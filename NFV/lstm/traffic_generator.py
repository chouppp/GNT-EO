#!/usr/bin/env python3
"""Expand existing vnr_flow_history.yaml logs into LSTM traffic sequences.

Rules:
1. For multiple records at the same time, use the last global_link_flow as the 100-unit baseline.
2. By default, read GNT/1 through GNT/8.
3. Use 17 baselines at times 400,500,...,2000; 400 is GNT warmup only, while plots show 500--2000.
4. Expand each window into 20 segments of five time units.
5. Mark the first seven (35%) as observed and the remaining 13 (65%) as prediction_target.
6. Add smooth perturbations around the baseline: small linear trends, periodic variation, and AR(1) noise.
7. Keep each segment within +/-15% of baseline and each link's 20-segment mean equal to baseline; zero-traffic links remain zero.
8. Use identical link indices and distinct random seeds across runs.
9. Before generating a run, remove its previous generated NPZ, complete YAML, and metadata files to prevent mixing stale results after interruption.

Outputs: GNT/<run>/lstm_traffic_500_2000.npz, lstm_traffic_500_2000.yaml, and lstm_traffic_500_2000_meta.yaml."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np
import yaml


Node = Hashable
Edge = Tuple[Node, Node]


RECORD_RE = re.compile(r"^- time:\s*(\S+)")
SOURCE_RE = re.compile(r"^  - source:\s*(\S+)")
TARGET_RE = re.compile(r"^    target:\s*(\S+)")
FLOW_RE = re.compile(r"^    flow:\s*(\S+)")


def parse_node(raw_value: str) -> Node:
    """Parse integer, float, or string node IDs commonly found in YAML."""
    value = raw_value.strip().strip("\"'")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def natural_sort_key(value: Node) -> Tuple[int, object]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def load_complete_edge_list(
    topology_path: Path | None,
    edge_reference_npz: Path | None,
) -> Tuple[List[Edge] | None, str]:
    """Read the complete fixed substrate-link set.
        Prefer p_net_directed.pkl; for tests or missing topology, use edge_sources/edge_targets from an existing NPZ.
        Return None if neither exists, allowing fallback to links present in traffic history."""
    if topology_path is not None and topology_path.exists():
        import pickle

        with topology_path.open("rb") as file:
            graph = pickle.load(file)
        if not hasattr(graph, "edges"):
            raise TypeError("topology 文件中不是有效的 NetworkX 图对象。")
        edges = [(source, target) for source, target in graph.edges()]
        return edges, f"topology:{topology_path}"

    if edge_reference_npz is not None and edge_reference_npz.exists():
        with np.load(edge_reference_npz) as reference:
            if (
                "edge_sources" not in reference
                or "edge_targets" not in reference
            ):
                raise KeyError(
                    "edge-reference-npz 缺少 edge_sources/edge_targets。"
                )
            edges = [
                (parse_node(str(source)), parse_node(str(target)))
                for source, target in zip(
                    reference["edge_sources"],
                    reference["edge_targets"],
                )
            ]
        return edges, f"npz:{edge_reference_npz}"

    return None, "history_union"


def parse_last_flow_record_per_time(
    history_path: Path,
) -> Dict[float, Dict[Edge, float]]:
    """Stream large history YAML files.
        global_link_flow is the last field in each record, so large vnr_flows need not be parsed.
        Overwrite duplicate times to retain the final record at each time."""
    if not history_path.exists():
        raise FileNotFoundError(f"找不到历史流量文件：{history_path}")

    last_record_by_time: Dict[float, Dict[Edge, float]] = {}
    current_time: float | None = None
    current_flow: Dict[Edge, float] = {}
    in_global_link_flow = False
    current_source: Node | None = None
    current_target: Node | None = None

    def finish_record() -> None:
        if current_time is not None:
            last_record_by_time[current_time] = dict(current_flow)

    with history_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = RECORD_RE.match(line)
            if match:
                finish_record()
                try:
                    current_time = float(match.group(1))
                except ValueError as error:
                    raise ValueError(
                        f"第 {line_number} 行的 time 无法转换为数值。"
                    ) from error
                current_flow = {}
                in_global_link_flow = False
                current_source = None
                current_target = None
                continue

            if line.startswith("  global_link_flow:"):
                in_global_link_flow = True
                current_source = None
                current_target = None
                continue

            if not in_global_link_flow:
                continue

            match = SOURCE_RE.match(line)
            if match:
                current_source = parse_node(match.group(1))
                current_target = None
                continue

            match = TARGET_RE.match(line)
            if match:
                current_target = parse_node(match.group(1))
                continue

            match = FLOW_RE.match(line)
            if match and current_source is not None and current_target is not None:
                try:
                    flow = max(float(match.group(1)), 0.0)
                except ValueError as error:
                    raise ValueError(
                        f"第 {line_number} 行的 flow 无法转换为数值。"
                    ) from error
                current_flow[(current_source, current_target)] = flow

    finish_record()

    if not last_record_by_time:
        raise ValueError("历史文件中没有找到有效的 records/time 流量记录。")

    return last_record_by_time


def build_baseline_arrays(
    records: Dict[float, Dict[Edge, float]],
    start_time: float,
    end_time: float,
    complete_edges: Sequence[Edge] | None = None,
) -> Tuple[np.ndarray, List[Edge], List[Node], np.ndarray]:
    """Convert history within the requested time range into consistent link vectors.
        time=100 represents (0,100], so select start_time <= time <= end_time.
        Extract links from the full history to retain temporarily inactive substrate links in fixed-size inputs."""
    selected_times = [
        time
        for time in sorted(records)
        if start_time <= time <= end_time
    ]
    if not selected_times:
        raise ValueError(
            f"历史文件中没有位于 [{start_time}, {end_time}] 的记录。"
        )

    base_times = np.asarray(selected_times, dtype=np.float64)
    if complete_edges is None:
        edge_set = {
            edge
            for flow_record in records.values()
            for edge in flow_record
        }
    else:
        edge_set = set(complete_edges)
        history_edges = {
            edge
            for flow_record in records.values()
            for edge in flow_record
        }
        missing_from_topology = history_edges - edge_set
        if missing_from_topology:
            raise ValueError(
                "历史流量中存在不属于完整底层链路集合的链路："
                f"{sorted(missing_from_topology)[:10]}"
            )
    edges = sorted(
        edge_set,
        key=lambda edge: (
            natural_sort_key(edge[0]),
            natural_sort_key(edge[1]),
        ),
    )
    nodes = sorted(
        {node for edge in edges for node in edge},
        key=natural_sort_key,
    )

    edge_index = {edge: index for index, edge in enumerate(edges)}
    baseline = np.zeros((len(base_times), len(edges)), dtype=np.float32)

    for window_id, base_time in enumerate(base_times):
        for edge, flow in records[float(base_time)].items():
            baseline[window_id, edge_index[edge]] = float(flow)

    return base_times, edges, nodes, baseline


def make_smooth_perturbation(
    rng: np.random.Generator,
    num_links: int,
    steps_per_window: int,
    trend_ratio: float,
    wave_ratio: float,
    link_noise_std: float,
    ar_coefficient: float,
    max_perturbation: float,
) -> np.ndarray:
    """Generate smooth, small perturbations within one window.
        Return multiplicative perturbation ratios shaped (steps_per_window, num_links)."""
    position = np.linspace(-1.0, 1.0, steps_per_window, dtype=np.float64)

    # Small shared trends and periodic changes preserve spatial correlation across links.
    slope = float(rng.uniform(-trend_ratio, trend_ratio))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    shared = (
        slope * position
        + wave_ratio
        * np.sin(np.linspace(0.0, 2.0 * math.pi, steps_per_window) + phase)
    )

    # Independent, temporally continuous AR(1) perturbations for each link.
    link_noise = np.zeros((steps_per_window, num_links), dtype=np.float64)
    stationary_scale = link_noise_std * math.sqrt(
        max(1.0 - ar_coefficient**2, 1e-12)
    )
    link_noise[0] = rng.normal(0.0, link_noise_std, size=num_links)
    for step in range(1, steps_per_window):
        link_noise[step] = (
            ar_coefficient * link_noise[step - 1]
            + rng.normal(0.0, stationary_scale, size=num_links)
        )

    raw_perturbation = shared[:, None] + link_noise

    # Project each link's perturbation onto these constraints:
    # 1) Zero mean over 20 segments.
    # 2) Every segment lies within [-max_perturbation, +max_perturbation].
    # Use bisection for each column's shift to enforce bounds more reliably than clipping then renormalizing.
    lower_shift = (
        raw_perturbation.min(axis=0, keepdims=True) - max_perturbation
    )
    upper_shift = (
        raw_perturbation.max(axis=0, keepdims=True) + max_perturbation
    )
    for _ in range(60):
        middle_shift = (lower_shift + upper_shift) / 2.0
        projected = np.clip(
            raw_perturbation - middle_shift,
            -max_perturbation,
            max_perturbation,
        )
        positive_mean = projected.mean(axis=0, keepdims=True) > 0.0
        lower_shift = np.where(
            positive_mean,
            middle_shift,
            lower_shift,
        )
        upper_shift = np.where(
            positive_mean,
            upper_shift,
            middle_shift,
        )

    perturbation = np.clip(
        raw_perturbation - (lower_shift + upper_shift) / 2.0,
        -max_perturbation,
        max_perturbation,
    )

    return perturbation.astype(np.float32)


def augment_baseline_flows(
    baseline: np.ndarray,
    steps_per_window: int,
    seed: int,
    trend_ratio: float,
    wave_ratio: float,
    link_noise_std: float,
    ar_coefficient: float,
    max_perturbation: float,
) -> np.ndarray:
    """Generate 20 segments around each historical baseline flow."""
    rng = np.random.default_rng(seed)
    num_windows, num_links = baseline.shape
    augmented = np.zeros(
        (num_windows, steps_per_window, num_links),
        dtype=np.float32,
    )

    for window_id in range(num_windows):
        perturbation = make_smooth_perturbation(
            rng=rng,
            num_links=num_links,
            steps_per_window=steps_per_window,
            trend_ratio=trend_ratio,
            wave_ratio=wave_ratio,
            link_noise_std=link_noise_std,
            ar_coefficient=ar_coefficient,
            max_perturbation=max_perturbation,
        )
        factors = 1.0 + perturbation
        augmented[window_id] = baseline[window_id][None, :] * factors

    return augmented


def build_time_arrays(
    base_times: np.ndarray,
    steps_per_window: int,
    window_duration: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The time=100 history record represents (0,100], with segment times 5,10,...,100."""
    substep_duration = window_duration / steps_per_window
    window_ends = base_times.astype(np.float32)
    window_starts = window_ends - window_duration
    offsets = np.arange(
        1,
        steps_per_window + 1,
        dtype=np.float32,
    ) * substep_duration
    times = window_starts[:, None] + offsets[None, :]
    return window_starts, window_ends, times


def build_traffic_matrices(
    link_flow: np.ndarray,
    edges: Sequence[Edge],
    nodes: Sequence[Node],
) -> np.ndarray:
    """Convert a link vector back into a node-by-node directed traffic matrix."""
    node_index = {node: index for index, node in enumerate(nodes)}
    matrices = np.zeros(
        (
            link_flow.shape[0],
            link_flow.shape[1],
            len(nodes),
            len(nodes),
        ),
        dtype=np.float32,
    )
    for edge_id, (source, target) in enumerate(edges):
        matrices[:, :, node_index[source], node_index[target]] = (
            link_flow[:, :, edge_id]
        )
    return matrices


def node_array(nodes: Sequence[Node]) -> np.ndarray:
    return np.asarray([str(node) for node in nodes], dtype=np.str_)


def edge_arrays(edges: Sequence[Edge]) -> Tuple[np.ndarray, np.ndarray]:
    sources = np.asarray([str(edge[0]) for edge in edges], dtype=np.str_)
    targets = np.asarray([str(edge[1]) for edge in edges], dtype=np.str_)
    return sources, targets


def save_npz(
    output_path: Path,
    baseline: np.ndarray,
    link_flow: np.ndarray,
    traffic_matrix: np.ndarray,
    base_times: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    times: np.ndarray,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    observed_steps: int,
) -> None:
    edge_sources, edge_targets = edge_arrays(edges)
    np.savez_compressed(
        output_path,
        baseline_link_flow=baseline,
        link_flow=link_flow,
        observed_link_flow=link_flow[:, :observed_steps, :],
        target_link_flow=link_flow[:, observed_steps:, :],
        traffic_matrix=traffic_matrix,
        observed_traffic_matrix=traffic_matrix[:, :observed_steps, :, :],
        target_traffic_matrix=traffic_matrix[:, observed_steps:, :, :],
        base_times=base_times.astype(np.float32),
        window_starts=window_starts,
        window_ends=window_ends,
        times=times,
        node_ids=node_array(nodes),
        edge_sources=edge_sources,
        edge_targets=edge_targets,
    )


def format_yaml_scalar(value: Node) -> str:
    if isinstance(value, str):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def save_full_yaml(
    output_path: Path,
    link_flow: np.ndarray,
    edges: Sequence[Edge],
    times: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    observed_steps: int,
) -> None:
    """Stream all 20 segments to avoid allocating a large YAML object in memory."""
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("records:\n")
        for window_id in range(link_flow.shape[0]):
            for step_id in range(link_flow.shape[1]):
                role = (
                    "observed"
                    if step_id < observed_steps
                    else "prediction_target"
                )
                file.write(f"- time: {float(times[window_id, step_id]):.6f}\n")
                file.write(
                    f"  base_time: {float(window_ends[window_id]):.6f}\n"
                )
                file.write(
                    f"  window_start: {float(window_starts[window_id]):.6f}\n"
                )
                file.write(
                    f"  window_end: {float(window_ends[window_id]):.6f}\n"
                )
                file.write(f"  substep: {step_id + 1}\n")
                file.write(f"  role: {role}\n")

                nonzero_ids = np.flatnonzero(link_flow[window_id, step_id] > 0)
                if len(nonzero_ids) == 0:
                    file.write("  global_link_flow: []\n")
                    continue

                file.write("  global_link_flow:\n")
                for edge_id in nonzero_ids:
                    source, target = edges[int(edge_id)]
                    flow = float(link_flow[window_id, step_id, edge_id])
                    file.write(
                        f"  - source: {format_yaml_scalar(source)}\n"
                    )
                    file.write(
                        f"    target: {format_yaml_scalar(target)}\n"
                    )
                    file.write(f"    flow: {flow:.8f}\n")


def save_metadata(
    output_path: Path,
    history_path: Path,
    base_times: np.ndarray,
    baseline: np.ndarray,
    link_flow: np.ndarray,
    traffic_matrix: np.ndarray,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    args: argparse.Namespace,
) -> None:
    mean_flow = link_flow.mean(axis=1)
    baseline_mae = float(np.mean(np.abs(mean_flow - baseline)))
    active_mask = baseline > 0
    if np.any(active_mask):
        relative_error = np.abs(mean_flow[active_mask] - baseline[active_mask])
        relative_error /= baseline[active_mask]
        baseline_mean_relative_error = float(relative_error.mean())
    else:
        baseline_mean_relative_error = 0.0

    metadata = {
        "description": "Traffic sequences augmented from VNR flow history",
        "history_file": str(history_path),
        "record_selection": "last record for each unique time",
        "edge_index_source": args.edge_index_source,
        "seed": args.seed,
        "node_count": len(nodes),
        "directed_link_count": len(edges),
        "num_windows": len(base_times),
        "requested_start_time": args.start_time,
        "requested_end_time": args.end_time,
        "first_base_time": float(base_times[0]),
        "last_base_time": float(base_times[-1]),
        "window_duration": args.window_duration,
        "steps_per_window": args.steps_per_window,
        "substep_duration": args.window_duration / args.steps_per_window,
        "observed_steps": args.observed_steps,
        "prediction_steps": args.steps_per_window - args.observed_steps,
        "observed_ratio": args.observed_steps / args.steps_per_window,
        "perturbation": {
            "trend_ratio": args.trend_ratio,
            "wave_ratio": args.wave_ratio,
            "link_noise_std": args.link_noise_std,
            "ar_coefficient": args.ar_coefficient,
            "max_perturbation": args.max_perturbation,
            "zero_baseline_remains_zero": True,
            "window_mean_preserves_baseline": True,
        },
        "validation": {
            "window_mean_baseline_mae": baseline_mae,
            "window_mean_baseline_mean_relative_error": (
                baseline_mean_relative_error
            ),
        },
        "array_shapes": {
            "baseline_link_flow": list(baseline.shape),
            "link_flow": list(link_flow.shape),
            "observed_link_flow": [
                link_flow.shape[0],
                args.observed_steps,
                link_flow.shape[2],
            ],
            "target_link_flow": [
                link_flow.shape[0],
                args.steps_per_window - args.observed_steps,
                link_flow.shape[2],
            ],
            "traffic_matrix": list(traffic_matrix.shape),
        },
        "links": [
            {
                "link_id": link_id,
                "source": source,
                "target": target,
            }
            for link_id, (source, target) in enumerate(edges)
        ],
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            metadata,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "批量读取GNT/1--8的VNR流量历史，"
            "为每100时间单位补出20个小段。"
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("GNT"),
        help="包含1、2、...实验子目录的根目录。",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[str(run) for run in range(1, 9)],
        help="需要处理的实验目录，默认：1 2 3 4 5 6 7 8。",
    )
    parser.add_argument(
        "--history-name",
        default="vnr_flow_history.yaml",
        help="每个实验目录中的历史流量文件名。",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("p_net_directed.pkl"),
        help="用于保留全部底层有向链路的NetworkX拓扑文件。",
    )
    parser.add_argument(
        "--edge-reference-npz",
        type=Path,
        default=None,
        help="拓扑文件不可用时，从已有NPZ读取固定链路顺序。",
    )
    parser.add_argument(
        "--npz-name",
        default="lstm_traffic_500_2000.npz",
        help="每个实验目录中生成的NPZ文件名。",
    )
    parser.add_argument(
        "--yaml-name",
        default="lstm_traffic_500_2000.yaml",
        help="每个实验目录中生成的完整YAML文件名。",
    )
    parser.add_argument(
        "--metadata-name",
        default="lstm_traffic_500_2000_meta.yaml",
        help="每个实验目录中生成的元数据文件名。",
    )
    parser.add_argument("--window-duration", type=float, default=100.0)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument(
        "--observed-steps",
        type=int,
        default=7,
        help="观测小段数量，默认7/20=35%。",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=400.0,
        help="生成区间起点。400为GNT计算预热时段，绘图从500开始。",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=2000.0,
        help="生成区间终点，包含该时刻的基准记录。",
    )
    parser.add_argument(
        "--trend-ratio",
        type=float,
        default=0.02,
        help="一个窗口内共享线性趋势的最大比例，默认0.02（较平滑）。",
    )
    parser.add_argument(
        "--wave-ratio",
        type=float,
        default=0.01,
        help="一个窗口内共享周期波动的幅度，默认0.01（较平滑）。",
    )
    parser.add_argument(
        "--link-noise-std",
        type=float,
        default=0.01,
        help="单链路AR平滑扰动的标准差，默认0.01（较平滑）。",
    )
    parser.add_argument(
        "--ar-coefficient",
        type=float,
        default=0.95,
        help="单链路AR(1)系数，默认0.95（增强相邻小段的连续性）。",
    )
    parser.add_argument(
        "--max-perturbation",
        type=float,
        default=0.03,
        help="单个小段相对基准值的最大扰动比例，默认0.03（3%）。",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.end_time <= args.start_time:
        raise ValueError("end-time 必须大于 start-time。")
    if args.window_duration <= 0:
        raise ValueError("window-duration 必须大于0。")
    if args.steps_per_window <= 1:
        raise ValueError("steps-per-window 必须大于1。")
    if not 0 < args.observed_steps < args.steps_per_window:
        raise ValueError("observed-steps 必须位于1和steps-per-window-1之间。")
    if not 0 <= args.ar_coefficient < 1:
        raise ValueError("ar-coefficient 必须位于[0,1)之间。")
    if not 0 <= args.max_perturbation < 1:
        raise ValueError("max-perturbation 必须位于[0,1)之间。")
    for name in (
        "trend_ratio",
        "wave_ratio",
        "link_noise_std",
        "max_perturbation",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能小于0。")


def remove_previous_outputs(
    run_dir: Path,
    history_path: Path,
    output_paths: Sequence[Path],
) -> None:
    """Remove previously generated files for the current run.
        Delete only explicitly named outputs directly under run_dir; protect traffic history, other runs, and unrelated user files."""
    resolved_run_dir = run_dir.resolve()
    resolved_history_path = history_path.resolve()

    for output_path in output_paths:
        resolved_output_path = output_path.resolve()

        if resolved_output_path.parent != resolved_run_dir:
            raise ValueError(
                "为避免误删，输出文件必须位于当前实验目录正下方："
                f"{output_path}"
            )

        if resolved_output_path == resolved_history_path:
            raise ValueError(
                "输出文件名不能与原始历史流量文件相同："
                f"{output_path}"
            )

        if not output_path.exists():
            continue

        if not output_path.is_file():
            raise IsADirectoryError(
                f"待清理的输出路径不是普通文件：{output_path}"
            )

        output_path.unlink()
        print(f"[CLEAN] 已删除上一轮生成文件：{output_path}")


def generate_one_run(
    run: str,
    run_index: int,
    records: Dict[float, Dict[Edge, float]],
    history_path: Path,
    complete_edges: Sequence[Edge],
    edge_index_source: str,
    args: argparse.Namespace,
) -> None:
    """Generate one LSTM traffic dataset from a historical run."""
    run_dir = args.base_dir / run
    run_dir.mkdir(parents=True, exist_ok=True)

    npz_output = run_dir / args.npz_name
    yaml_output = run_dir / args.yaml_name
    metadata_output = run_dir / args.metadata_name

    remove_previous_outputs(
        run_dir=run_dir,
        history_path=history_path,
        output_paths=(npz_output, yaml_output, metadata_output),
    )

    run_seed = args.seed + run_index
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = run_seed
    run_args.edge_index_source = edge_index_source

    base_times, edges, nodes, baseline = build_baseline_arrays(
        records,
        start_time=args.start_time,
        end_time=args.end_time,
        complete_edges=complete_edges,
    )
    link_flow = augment_baseline_flows(
        baseline=baseline,
        steps_per_window=args.steps_per_window,
        seed=run_seed,
        trend_ratio=args.trend_ratio,
        wave_ratio=args.wave_ratio,
        link_noise_std=args.link_noise_std,
        ar_coefficient=args.ar_coefficient,
        max_perturbation=args.max_perturbation,
    )
    window_starts, window_ends, times = build_time_arrays(
        base_times=base_times,
        steps_per_window=args.steps_per_window,
        window_duration=args.window_duration,
    )
    traffic_matrix = build_traffic_matrices(
        link_flow=link_flow,
        edges=edges,
        nodes=nodes,
    )

    save_npz(
        output_path=npz_output,
        baseline=baseline,
        link_flow=link_flow,
        traffic_matrix=traffic_matrix,
        base_times=base_times,
        window_starts=window_starts,
        window_ends=window_ends,
        times=times,
        nodes=nodes,
        edges=edges,
        observed_steps=args.observed_steps,
    )
    save_full_yaml(
        output_path=yaml_output,
        link_flow=link_flow,
        edges=edges,
        times=times,
        window_starts=window_starts,
        window_ends=window_ends,
        observed_steps=args.observed_steps,
    )
    save_metadata(
        output_path=metadata_output,
        history_path=history_path,
        base_times=base_times,
        baseline=baseline,
        link_flow=link_flow,
        traffic_matrix=traffic_matrix,
        nodes=nodes,
        edges=edges,
        args=run_args,
    )

    window_mean = link_flow.mean(axis=1)
    mean_mae = float(np.mean(np.abs(window_mean - baseline)))
    active_mask = baseline > 0
    relative_changes = np.zeros_like(link_flow)
    np.divide(
        np.abs(link_flow - baseline[:, None, :]),
        baseline[:, None, :],
        out=relative_changes,
        where=baseline[:, None, :] > 0,
    )
    max_relative_change = float(
        relative_changes[
            np.broadcast_to(active_mask[:, None, :], link_flow.shape)
        ].max(initial=0.0)
    )

    print(f"\n[Run {run}] 历史流量补全完成")
    print(f"生成时间范围：{args.start_time:g}--{args.end_time:g}")
    print(f"大时段数量：{len(base_times)}")
    print(f"节点数：{len(nodes)}")
    print(f"有向链路数：{len(edges)}")
    print(f"链路索引来源：{edge_index_source}")
    print(
        "完整流量形状："
        f"{tuple(link_flow.shape)} "
        "(大时段, 20个小段, 有向链路)"
    )
    print(
        "LSTM输入形状："
        f"{tuple(link_flow[:, :args.observed_steps, :].shape)}"
    )
    print(
        "LSTM目标形状："
        f"{tuple(link_flow[:, args.observed_steps:, :].shape)}"
    )
    print(f"20段均值与原始基准的MAE：{mean_mae:.10f}")
    print(f"最大相对扰动：{max_relative_change:.4%}")
    print(f"随机种子：{run_seed}")
    print(f"NPZ：{npz_output}")
    print(f"完整YAML：{yaml_output}")
    print(f"元数据：{metadata_output}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    records_by_run: Dict[str, Dict[float, Dict[Edge, float]]] = {}
    history_paths: Dict[str, Path] = {}
    for run in args.runs:
        history_path = args.base_dir / run / args.history_name
        history_paths[run] = history_path
        records_by_run[run] = parse_last_flow_record_per_time(history_path)

    complete_edges, edge_index_source = load_complete_edge_list(
        topology_path=args.topology,
        edge_reference_npz=args.edge_reference_npz,
    )

    # Without a shared topology file, use the union of links from all eight histories.
    # Ensure consistent feature order and dimensionality across LSTM datasets.
    if complete_edges is None:
        edge_set = {
            edge
            for records in records_by_run.values()
            for flow_record in records.values()
            for edge in flow_record
        }
        complete_edges = sorted(
            edge_set,
            key=lambda edge: (
                natural_sort_key(edge[0]),
                natural_sort_key(edge[1]),
            ),
        )
        edge_index_source = "history_union_all_runs"

    for run_index, run in enumerate(args.runs):
        generate_one_run(
            run=run,
            run_index=run_index,
            records=records_by_run[run],
            history_path=history_paths[run],
            complete_edges=complete_edges,
            edge_index_source=edge_index_source,
            args=args,
        )

    print(f"\n全部完成：共生成 {len(args.runs)} 份LSTM预测数据。")


if __name__ == "__main__":
    main()