> 命令中的项目路径相对于项目根目录；先激活已安装所需依赖的 Python 环境。

# EO node-only A3C-GCN training

This module trains only the physical-node selector.  For every completed node
mapping, the environment performs EO-compatible directed link lookup using the
candidate paths loaded from `link_table` and the residual directed bandwidth
loaded from `link_single_table`.

## Safety boundary

The CLI opens MySQL only once and reads a `DatabaseSnapshot`.  It does not load
the complete `link_table`; `database.max_paths_per_pair` limits each directed
endpoint pair to its shortest K candidates. Every A3C worker
owns an in-memory copy and performs allocation/rollback there.  Training never
updates `node_single_table`, `link_single_table`, or `link_table`.

## Files

- `training_env.py`: event lifecycle, directed action mask, EO path selection,
  in-memory allocation and release, and reward calculation.
- `training_model.py`: direction-aware 11-feature Actor-Critic.
- `shared_optim.py`: process-shared Adam optimizer.
- `trainer.py`: asynchronous CPU workers, evaluation, logs and checkpoints.
- `train.py`: command-line entrypoint.
- `train_config.example.yaml`: runnable configuration template.

## Dependencies

Install PyTorch for the intended Python/CUDA version first, then install:

```text
torch-geometric
pymysql
PyYAML
numpy
```

Use a Python environment with the required PyTorch and PyG dependencies installed.

## Run

Copy `train_config.example.yaml`, add disjoint training and validation event
files, and expose the database password without writing it into source code.

PowerShell:

```powershell
$env:EO_DB_PASSWORD = "your-password"
python -m NFV.a3c_gcn.train --config NFV/a3c_gcn/train_config.example.yaml
```

The output directory receives:

- `a3c_gcn_full_last.pt`: Actor, Critic, optimizer, and config for resuming.
- `a3c_gcn_actor_last.pt`: node-selector weights for inference.
- `a3c_gcn_full_best.pt` / `a3c_gcn_actor_best.pt`: best periodic candidate
  selected by deterministic greedy evaluation. The full best file is intended
  for evaluation/inference and does not contain a resumable optimizer state.
- `selection_candidates/<run-id>/candidate_*.pt`: periodic model candidates.
- `checkpoint_evaluations.json`: greedy result for every candidate.
- `worker_*.jsonl`: per-episode training metrics.
- `training_greedy_evaluation.json`: deterministic argmax replay over the
  training traces, including acceptance and failure-reason counts.
- `training_summary.json`: aggregate training metrics, the best sampled
  episode, the in-sample greedy replay, and validation result.

`overall_acceptance_ratio` is collected while A3C samples actions during
training. `training_greedy_evaluation.acceptance_ratio` is collected after
training by replaying `event_files` with argmax actions. The latter is an
in-sample diagnostic and must not be reported as an independent test result.
Configure disjoint `validation_event_files` for that purpose.

`checkpoint_interval` controls candidate frequency. Model selection uses
`validation_event_files` when configured; otherwise it falls back to
`event_files` and records `model_selection_in_sample: true`. For final
experiments, always provide disjoint validation traces.

The example currently points to the single existing event trace.  It is useful
for integration training only; final experiments should use multiple disjoint
event traces and VNR IDs.
