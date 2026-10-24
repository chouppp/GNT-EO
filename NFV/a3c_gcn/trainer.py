"""Asynchronous A3C trainer for EO's node-only GCN policy."""

from __future__ import annotations

import copy
import json
import os
import random
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .training_env import (
    DatabaseSnapshot,
    EOTrainingEnvironment,
    load_events,
    load_vnet,
    resolve_vnet_path,
)


@dataclass
class TrainConfig:
    vnet_dir: str
    event_files: List[str]
    output_dir: str
    validation_event_files: Optional[List[str]] = None
    filename_pattern: str = "vinre0.5_node_{vnr_id}.yaml"
    episodes: int = 30
    num_workers: int = 4
    embedding_dim: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    distance_limit: Optional[float] = 10.0
    reusable: bool = False
    intermediate_reward: float = 0.01
    success_reward: float = 1.0
    failure_reward: float = -1.0
    quality_weight: float = 0.5
    progress_interval: int = 50
    evaluate_training_greedy: bool = True
    checkpoint_interval: int = 10
    seed: int = 2026

    def __post_init__(self):
        from NFV.project_paths import resolve_config_paths
        resolved = resolve_config_paths(vars(self))
        for key in ("vnet_dir", "event_files", "output_dir", "validation_event_files"):
            setattr(self, key, resolved[key])


def _require_torch():
    try:
        import torch
        import torch.multiprocessing as mp
        from torch.distributions import Categorical
        from torch_geometric.data import Data
        from .training_model import DirectionalActorCritic
    except ImportError as exc:
        raise ImportError(
            "Training requires torch and torch_geometric; see TRAINING.md."
        ) from exc
    return torch, mp, Categorical, Data, DirectionalActorCritic


def _make_env(snapshot: DatabaseSnapshot, config: TrainConfig) -> EOTrainingEnvironment:
    return EOTrainingEnvironment(
        snapshot=snapshot,
        distance_limit=config.distance_limit,
        reusable=config.reusable,
        intermediate_reward=config.intermediate_reward,
        success_reward=config.success_reward,
        failure_reward=config.failure_reward,
        quality_weight=config.quality_weight,
    )


def _to_data(observation, Data, torch):
    data = Data(
        x=torch.as_tensor(observation.x, dtype=torch.float32),
        edge_index=torch.as_tensor(observation.edge_index, dtype=torch.long),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def _update_global(local_model, global_model, optimizer, trajectory, config, torch):
    if not trajectory:
        return None
    rewards = [item[3] for item in trajectory]
    returns = []
    discounted = 0.0
    for reward in reversed(rewards):
        discounted = float(reward) + config.gamma * discounted
        returns.insert(0, discounted)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    log_probs = torch.stack([item[0] for item in trajectory])
    values = torch.stack([item[1] for item in trajectory]).reshape(-1)
    entropies = torch.stack([item[2] for item in trajectory])
    advantages = returns - values
    actor_loss = -(log_probs * advantages.detach()).mean() - config.entropy_coef * entropies.mean()
    critic_loss = 0.5 * advantages.pow(2).mean()
    loss = actor_loss + config.value_loss_coef * critic_loss

    optimizer.zero_grad()
    local_model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(local_model.parameters(), config.max_grad_norm)
    for local_parameter, global_parameter in zip(local_model.parameters(), global_model.parameters()):
        if local_parameter.grad is not None:
            global_parameter._grad = local_parameter.grad.detach().clone()
    optimizer.step()
    local_model.load_state_dict(global_model.state_dict())
    return float(loss.detach())


def _claim_episode(counter, total_episodes):
    with counter.get_lock():
        if counter.value >= total_episodes:
            return None
        episode = counter.value
        counter.value += 1
        return episode


def _worker(
    worker_id,
    global_model,
    optimizer,
    snapshot,
    config,
    episode_counter,
    stats,
    candidate_dir,
):
    torch, _, Categorical, Data, DirectionalActorCritic = _require_torch()
    torch.set_num_threads(1)
    seed = config.seed + worker_id * 1009
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    local_model = DirectionalActorCritic(11, config.embedding_dim)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()
    env = _make_env(snapshot, config)
    log_path = Path(config.output_dir) / f"worker_{worker_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_stream:
        while True:
            episode = _claim_episode(episode_counter, config.episodes)
            if episode is None:
                break
            event_path = config.event_files[episode % len(config.event_files)]
            env.reset()
            print(
                f"[worker {worker_id}] episode {episode + 1}/{config.episodes} "
                f"using {event_path}",
                flush=True,
            )
            episode_requests = episode_successes = 0
            episode_reward = 0.0
            losses = []
            for event in load_events(event_path):
                vnr_id = int(event["v_net_id"])
                if int(event["type"]) == 0:
                    env.release(vnr_id)
                    continue
                vnet_path = resolve_vnet_path(config.vnet_dir, config.filename_pattern, vnr_id)
                if not os.path.exists(vnet_path):
                    continue
                virtual_nodes, virtual_edges = load_vnet(vnet_path)
                observation = env.begin_request(vnr_id, virtual_nodes, virtual_edges)
                if observation is None:
                    continue
                episode_requests += 1
                trajectory = []
                terminal_info = {"success": False}
                while observation is not None:
                    if not observation.action_mask.any():
                        _, terminal_reward, _, terminal_info = env.fail_current_request(
                            "no feasible action"
                        )
                        if trajectory:
                            previous = trajectory[-1]
                            trajectory[-1] = (*previous[:3], terminal_reward)
                        break
                    data = _to_data(observation, Data, torch)
                    logits = local_model.act(data).reshape(-1)
                    value = local_model.evaluate(data).reshape(())
                    mask = torch.as_tensor(observation.action_mask, dtype=torch.bool)
                    distribution = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
                    action_index = distribution.sample()
                    physical_node = observation.node_ids[int(action_index)]
                    next_observation, reward, done, terminal_info = env.step(physical_node)
                    trajectory.append(
                        (distribution.log_prob(action_index), value, distribution.entropy(), reward)
                    )
                    observation = next_observation
                    if done:
                        break
                loss = _update_global(
                    local_model, global_model, optimizer, trajectory, config, torch
                )
                if loss is not None:
                    losses.append(loss)
                final_reward = trajectory[-1][3] if trajectory else config.failure_reward
                episode_reward += float(final_reward)
                episode_successes += int(bool(terminal_info.get("success")))
                if config.progress_interval > 0 and episode_requests % config.progress_interval == 0:
                    print(
                        f"[worker {worker_id}] episode {episode + 1}: "
                        f"requests={episode_requests}, successes={episode_successes}, "
                        f"acceptance={episode_successes / max(episode_requests, 1):.3f}",
                        flush=True,
                    )

            with stats.get_lock():
                stats[0] += episode_requests
                stats[1] += episode_successes
                stats[2] += episode_reward
                stats[3] += 1
                completed_episodes = int(stats[3])
            if (
                config.checkpoint_interval > 0
                and completed_episodes % config.checkpoint_interval == 0
            ):
                candidate_path = Path(candidate_dir) / (
                    f"candidate_{completed_episodes:06d}_"
                    f"episode_{episode + 1:06d}_worker_{worker_id}.pt"
                )
                torch.save({
                    "model_state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in local_model.state_dict().items()
                    },
                    "completed_episodes": completed_episodes,
                    "episode_index": episode + 1,
                    "worker": worker_id,
                }, candidate_path)
            log_stream.write(json.dumps({
                "episode": episode,
                "worker": worker_id,
                "requests": episode_requests,
                "successes": episode_successes,
                "acceptance_ratio": episode_successes / max(episode_requests, 1),
                "reward": episode_reward,
                "mean_loss": float(np.mean(losses)) if losses else None,
            }, ensure_ascii=False) + "\n")
            log_stream.flush()
            print(
                f"[worker {worker_id}] episode {episode + 1} finished: "
                f"requests={episode_requests}, successes={episode_successes}",
                flush=True,
            )


def evaluate(model, snapshot, config: TrainConfig, event_files: Sequence[str]):
    torch, _, _, Data, _ = _require_torch()
    model.eval()
    total = successes = 0
    total_reward = 0.0
    failure_reasons = Counter()
    with torch.no_grad():
        for event_path in event_files:
            env = _make_env(snapshot, config)
            for event in load_events(event_path):
                vnr_id = int(event["v_net_id"])
                if int(event["type"]) == 0:
                    env.release(vnr_id)
                    continue
                vnet_path = resolve_vnet_path(config.vnet_dir, config.filename_pattern, vnr_id)
                if not os.path.exists(vnet_path):
                    continue
                nodes, edges = load_vnet(vnet_path)
                observation = env.begin_request(vnr_id, nodes, edges)
                if observation is None:
                    continue
                total += 1
                info = {"success": False}
                reward = config.failure_reward
                while observation is not None:
                    if not observation.action_mask.any():
                        _, reward, _, info = env.fail_current_request("no feasible action")
                        break
                    data = _to_data(observation, Data, torch)
                    logits = model.act(data).reshape(-1)
                    mask = torch.as_tensor(observation.action_mask, dtype=torch.bool)
                    action_index = int(logits.masked_fill(~mask, -torch.inf).argmax())
                    observation, reward, done, info = env.step(
                        observation.node_ids[action_index]
                    )
                    if done:
                        break
                successes += int(bool(info.get("success")))
                if not info.get("success"):
                    failure_reasons[str(info.get("reason", "unknown failure"))] += 1
                total_reward += float(reward)
    model.train()
    return {
        "requests": total,
        "successes": successes,
        "acceptance_ratio": successes / max(total, 1),
        "mean_terminal_reward": total_reward / max(total, 1),
        "action_selection": "greedy_argmax",
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }


def _read_episode_records(output_dir: Path, num_workers: int):
    records = []
    for worker_id in range(max(1, num_workers)):
        log_path = output_dir / f"worker_{worker_id}.jsonl"
        if not log_path.exists():
            continue
        with open(log_path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def train(snapshot: DatabaseSnapshot, config: TrainConfig):
    if not config.event_files:
        raise ValueError("event_files must contain at least one training trace")
    torch, mp, _, _, DirectionalActorCritic = _require_torch()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "selection_candidates" / uuid.uuid4().hex
    candidate_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    global_model = DirectionalActorCritic(11, config.embedding_dim)
    global_model.share_memory()
    from .shared_optim import SharedAdam

    optimizer = SharedAdam(global_model.parameters(), lr=config.learning_rate)
    optimizer.share_memory()

    context = mp.get_context("spawn")
    episode_counter = context.Value("i", 0)
    stats = context.Array("d", [0.0, 0.0, 0.0, 0.0], lock=True)
    processes = []
    for worker_id in range(max(1, config.num_workers)):
        process = context.Process(
            target=_worker,
            args=(
                worker_id,
                global_model,
                optimizer,
                snapshot,
                config,
                episode_counter,
                stats,
                str(candidate_dir),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"A3C worker failed with exit code {process.exitcode}")

    last_model_state = {
        name: value.detach().cpu().clone()
        for name, value in global_model.state_dict().items()
    }
    checkpoint = {
        "model_state_dict": last_model_state,
        "actor_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in global_model.actor.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "a3c_gcn_full_last.pt")
    torch.save(global_model.actor.state_dict(), output_dir / "a3c_gcn_actor_last.pt")
    training_greedy_evaluation = None
    if config.evaluate_training_greedy:
        print(
            "[evaluation] replaying training traces with deterministic greedy actions",
            flush=True,
        )
        training_greedy_evaluation = evaluate(
            global_model, snapshot, config, config.event_files
        )
        training_greedy_evaluation.update({
            "dataset": "training_event_files",
            "in_sample": True,
        })
        with open(
            output_dir / "training_greedy_evaluation.json", "w", encoding="utf-8"
        ) as stream:
            json.dump(
                training_greedy_evaluation, stream, ensure_ascii=False, indent=2
            )
        print(
            "[evaluation] training greedy replay: "
            f"requests={training_greedy_evaluation['requests']}, "
            f"successes={training_greedy_evaluation['successes']}, "
            f"acceptance={training_greedy_evaluation['acceptance_ratio']:.3f}",
            flush=True,
        )
    validation_files = config.validation_event_files or []
    validation = evaluate(global_model, snapshot, config, validation_files) if validation_files else None
    if validation is not None:
        validation.update({
            "dataset": "validation_event_files",
            "in_sample": False,
        })
    selection_files = validation_files or config.event_files
    selection_dataset = (
        "validation_event_files" if validation_files else "training_event_files"
    )
    selection_is_in_sample = not bool(validation_files)
    candidate_paths = sorted(candidate_dir.glob("candidate_*.pt"))
    candidates = []
    for candidate_path in candidate_paths:
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        candidates.append((candidate_path, candidate))
    candidates.append((output_dir / "a3c_gcn_full_last.pt", checkpoint))

    print(
        f"[model-selection] evaluating {len(candidates)} checkpoints on "
        f"{selection_dataset}",
        flush=True,
    )
    best_candidate = None
    best_selection = None
    checkpoint_evaluations = []
    for candidate_path, candidate in candidates:
        global_model.load_state_dict(candidate["model_state_dict"])
        if (
            candidate_path.name == "a3c_gcn_full_last.pt"
            and not validation_files
            and training_greedy_evaluation is not None
        ):
            result = dict(training_greedy_evaluation)
        else:
            result = evaluate(global_model, snapshot, config, selection_files)
        result.update({
            "checkpoint": str(candidate_path),
            "completed_episodes": candidate.get("completed_episodes", config.episodes),
            "source_episode_index": candidate.get("episode_index", config.episodes),
        })
        checkpoint_evaluations.append(result)
        print(
            f"[model-selection] {candidate_path.name}: "
            f"successes={result['successes']}/{result['requests']}, "
            f"acceptance={result['acceptance_ratio']:.3f}",
            flush=True,
        )
        score = (
            float(result["acceptance_ratio"]),
            int(result["successes"]),
            float(result["mean_terminal_reward"]),
        )
        if best_selection is None or score > best_selection[0]:
            best_selection = (score, result)
            best_candidate = candidate

    global_model.load_state_dict(best_candidate["model_state_dict"])
    best_model_evaluation = dict(best_selection[1])
    best_model_evaluation.update({
        "dataset": selection_dataset,
        "in_sample": selection_is_in_sample,
    })
    best_checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in global_model.state_dict().items()
        },
        "actor_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in global_model.actor.state_dict().items()
        },
        "optimizer_state_dict": None,
        "config": asdict(config),
        "selection_evaluation": best_model_evaluation,
        "resumable": False,
    }
    torch.save(best_checkpoint, output_dir / "a3c_gcn_full_best.pt")
    torch.save(global_model.actor.state_dict(), output_dir / "a3c_gcn_actor_best.pt")
    with open(
        output_dir / "checkpoint_evaluations.json", "w", encoding="utf-8"
    ) as stream:
        json.dump(checkpoint_evaluations, stream, ensure_ascii=False, indent=2)
    print(
        "[model-selection] best checkpoint: "
        f"{Path(best_model_evaluation['checkpoint']).name}, "
        f"acceptance={best_model_evaluation['acceptance_ratio']:.3f}",
        flush=True,
    )

    global_model.load_state_dict(last_model_state)
    episode_records = _read_episode_records(output_dir, config.num_workers)
    best_episode = (
        max(
            episode_records,
            key=lambda record: (
                float(record["acceptance_ratio"]),
                float(record["reward"]),
                -int(record["episode"]),
            ),
        )
        if episode_records
        else None
    )
    if best_episode is not None:
        best_episode = dict(best_episode)
        best_episode["episode_index"] = int(best_episode.pop("episode")) + 1
    overall_acceptance = float(stats[1] / max(stats[0], 1.0))
    summary = {
        "episodes": int(stats[3]),
        "requests": int(stats[0]),
        "successes": int(stats[1]),
        "acceptance_ratio": overall_acceptance,
        "overall_acceptance_ratio": overall_acceptance,
        "mean_episode_acceptance_ratio": (
            float(np.mean([record["acceptance_ratio"] for record in episode_records]))
            if episode_records
            else 0.0
        ),
        "best_episode": best_episode,
        "mean_episode_reward": float(stats[2] / max(stats[3], 1.0)),
        "training_greedy_evaluation": training_greedy_evaluation,
        "best_model_evaluation": best_model_evaluation,
        "model_selection_dataset": selection_dataset,
        "model_selection_in_sample": selection_is_in_sample,
        "validation": validation,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
