#!/usr/bin/env python3
"""Predict raw-log GNT averaged across eight runs by default; recorded and window modes remain available.

Recorded mode uses a leave-one-run-out contemporaneous mean, adding LSTM residuals only when independent validation improves substantially. Window mode predicts mean link traffic over the last 13 synthetic segments from the first seven, then computes temperature.

Default data: GNT/1 through GNT/8, each containing lstm_traffic_500_2000.npz.

Leave-one-run-out protocol:
1. Hold out one run and train on the other seven.
2. Treat each link as a time-series sample instead of fitting 226 outputs from very few samples.
3. Split internal validation by whole run to avoid leakage between link samples.
4. Observe the first seven segments (35%) of each 100-unit window and predict the mean of the remaining 13.
5. Combine the predicted remaining mean with the seven observed segments.
6. Average over all 20 segments to obtain link traffic for each 100-unit window.
7. Compute true/predicted GNT from adjacent windows, clipping each domain before averaging.
8. Learn residuals relative to the last observation and report last-value/observed-mean baselines.

This window GNT target comes from synthetic segments, not event GNT in resource_usage.
Per-run outputs: lstm_prediction_500_2000.npz, lstm_prediction_gnt_500_2000.yaml, lstm_training_history.yaml, lstm_gnt_prediction_comparison.png.
Root outputs: lstm_gnt_prediction_summary.yaml and lstm_gnt_prediction_mean_comparison.png."""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml


DOMAINS = [
    [7, 2, 4, 5, 6],
    [18, 1, 3, 8, 10],
    [20, 9, 15, 17, 19],
    [12, 0, 21, 22, 23],
    [11, 14, 24, 13, 16],
]

GNT_CLIP_MIN = -10000.0
GNT_CLIP_MAX = 10000.0
ENTROPY_EPS = 1e-4
TEMPERATURE_TARGET = "adjacent_window_mean_flow"


@dataclass
class RunData:
    run: str
    path: Path
    observed: np.ndarray
    target: np.ndarray
    complete: np.ndarray
    base_times: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray


def require_tensorflow():
    """Import TensorFlow lazily so data and GNT functions can be tested independently."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "未安装TensorFlow。请先执行：pip install tensorflow"
        ) from error
    return tf


def parse_node(value: object) -> int | float | str:
    text = str(value)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def load_run_data(path: Path, run: str) -> RunData:
    if not path.exists():
        raise FileNotFoundError(f"找不到第{run}次实验数据：{path}")

    with np.load(path) as data:
        required = {
            "link_flow",
            "observed_link_flow",
            "target_link_flow",
            "base_times",
            "edge_sources",
            "edge_targets",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} 缺少字段：{sorted(missing)}")

        complete = np.asarray(data["link_flow"], dtype=np.float32)
        observed = np.asarray(data["observed_link_flow"], dtype=np.float32)
        target = np.asarray(data["target_link_flow"], dtype=np.float32)
        base_times = np.asarray(data["base_times"], dtype=np.float32)
        edge_sources = np.asarray(data["edge_sources"]).astype(str)
        edge_targets = np.asarray(data["edge_targets"]).astype(str)

    if any(array.ndim != 3 for array in (complete, observed, target)):
        raise ValueError(f"{path} 中流量数组必须为三维数组。")
    if any(not np.all(np.isfinite(array)) or np.any(array < 0)
           for array in (complete, observed, target)):
        raise ValueError(f"{path} 中流量必须非负且有限。")
    if min(complete.shape) == 0 or observed.shape[1] == 0 or target.shape[1] == 0:
        raise ValueError(f"{path} 中流量数组不能为空。")
    if observed.shape[0] != complete.shape[0]:
        raise ValueError(f"{path} 的时段数量不一致。")
    if target.shape[0] != complete.shape[0]:
        raise ValueError(f"{path} 的目标时段数量不一致。")
    if observed.shape[2] != complete.shape[2]:
        raise ValueError(f"{path} 的输入链路维度不一致。")
    if target.shape[2] != complete.shape[2]:
        raise ValueError(f"{path} 的目标链路维度不一致。")
    if observed.shape[1] + target.shape[1] != complete.shape[1]:
        raise ValueError(
            f"{path} 的观测段数{observed.shape[1]}和"
            f"预测段数{target.shape[1]}无法组成"
            f"{complete.shape[1]}段完整序列。"
        )
    if len(base_times) != complete.shape[0]:
        raise ValueError(f"{path} 的base_times数量不正确。")
    if (base_times.ndim != 1 or not np.all(np.isfinite(base_times))
            or len(base_times) < 2 or np.any(np.diff(base_times) <= 0)):
        raise ValueError(f"{path} 的base_times必须严格递增且至少含两个窗口。")
    if not np.allclose(np.diff(base_times), 100.0):
        raise ValueError(f"{path} 必须包含连续的100单位窗口，不能跨缺失窗口差分。")
    if (len(edge_sources) != complete.shape[2] or len(edge_targets) != complete.shape[2]
            or len(set(zip(edge_sources, edge_targets))) != complete.shape[2]):
        raise ValueError(f"{path} 的链路索引长度错误或重复。")
    if not np.array_equal(observed, complete[:, :observed.shape[1], :]):
        raise ValueError(f"{path} 的观测数据不是完整序列的前缀。")
    if not np.array_equal(target, complete[:, observed.shape[1]:, :]):
        raise ValueError(f"{path} 的目标数据不是完整序列的后缀。")

    return RunData(
        run=run,
        path=path,
        observed=observed,
        target=target,
        complete=complete,
        base_times=base_times,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
    )


def load_all_runs(
    base_dir: Path,
    runs: Sequence[str],
    input_name: str,
) -> Dict[str, RunData]:
    datasets = {
        run: load_run_data(base_dir / run / input_name, run)
        for run in runs
    }

    reference = datasets[runs[0]]
    for run in runs[1:]:
        current = datasets[run]
        if not np.array_equal(current.base_times, reference.base_times):
            raise ValueError(f"第{run}次实验的时间轴不一致。")
        if not np.array_equal(current.edge_sources, reference.edge_sources):
            raise ValueError(f"第{run}次实验的edge_sources顺序不一致。")
        if not np.array_equal(current.edge_targets, reference.edge_targets):
            raise ValueError(f"第{run}次实验的edge_targets顺序不一致。")
        if current.observed.shape[1:] != reference.observed.shape[1:]:
            raise ValueError(f"第{run}次实验的LSTM输入形状不一致。")
        if current.target.shape[1:] != reference.target.shape[1:]:
            raise ValueError(f"第{run}次实验的LSTM目标形状不一致。")

    return datasets


def average_run_data(datasets: Dict[str, RunData], runs: Sequence[str]) -> RunData:
    """First average each window, segment, and link across eight runs."""
    if not runs:
        raise ValueError("平均数据至少需要一组实验。")
    reference = datasets[runs[0]]
    return RunData(
        run="mean_of_8",
        path=Path("average_of_runs") / "_".join(runs),
        observed=np.mean([datasets[r].observed for r in runs], axis=0).astype(np.float32),
        target=np.mean([datasets[r].target for r in runs], axis=0).astype(np.float32),
        complete=np.mean([datasets[r].complete for r in runs], axis=0).astype(np.float32),
        base_times=reference.base_times.copy(),
        edge_sources=reference.edge_sources.copy(),
        edge_targets=reference.edge_targets.copy(),
    )


def run_averaged_window_fit(tf, datasets, args, output_dir, training_sets=None):
    """Average the current eight datasets; optionally train on generated data and validate on the current mean sample."""
    averaged = average_run_data(datasets, args.runs)
    is_validation = training_sets is not None
    if is_validation:
        model, history, sample_count, info = train_generated_sets(
            tf, training_sets, averaged, args,
        )
    else:
        model, history, sample_count, info = train_all_runs_in_sample(
            tf, [averaged], args,
        )
    info.update({
        "source_runs": list(args.runs),
        "averaging_order": "average 8 runs per window/substep/link before model training",
        "evaluation_scope": ("current averaged sample used only for validation and reporting"
                             if is_validation else
                             "same averaged samples used for training"),
    })
    predicted_target = predict_future_mean(
        model, averaged.observed, averaged.target.shape[1], args.batch_size,
    )
    metrics, actual_flow, predicted_flow, actual_gnt, predicted_gnt = evaluate_prediction(
        averaged, predicted_target, args.evaluation_start,
    )
    baselines = {}
    baseline_gnt = {}
    for name, forecast in baseline_predictions(averaged.observed, averaged.target.shape[1]).items():
        result = evaluate_prediction(averaged, forecast, args.evaluation_start)
        baselines[name] = result[0]
        baseline_gnt[name] = gnt_values(result[4])
    metrics["baselines"] = baselines
    run_dir = output_dir / "mean_of_8"
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / args.prediction_npz_name
    yaml_path = run_dir / args.prediction_yaml_name
    save_prediction_npz(npz_path, averaged, predicted_target,
                        np.concatenate([averaged.observed, predicted_target], axis=1),
                        predicted_flow, actual_flow)
    save_prediction_yaml(yaml_path, averaged, predicted_flow, actual_flow,
                         predicted_gnt, actual_gnt, metrics, args.runs, sample_count, info)
    save_training_history(run_dir / args.history_name, history)
    if args.save_models:
        model.save(run_dir / "lstm_traffic_model.keras")
    plot_run_gnt_comparison(run_dir / args.run_figure_name, averaged.run,
                            averaged.base_times, predicted_gnt, actual_gnt, args)
    actual_values = gnt_values(actual_gnt)[None, :]
    predicted_values = gnt_values(predicted_gnt)[None, :]
    mask = averaged.base_times >= args.evaluation_start
    mask[0] = False
    comparison = {
        "lstm": {"per_run_time": error_metrics(predicted_values[:, mask], actual_values[:, mask]),
                 "mean_curve": error_metrics(predicted_values[:, mask].mean(0), actual_values[:, mask].mean(0))},
        **{name: {"per_run_time": error_metrics(values[None, mask], actual_values[:, mask]),
                  "mean_curve": error_metrics(values[None, mask].mean(0), actual_values[:, mask].mean(0))}
           for name, values in baseline_gnt.items()},
    }
    figure_path = output_dir / args.summary_figure_name
    plot_summary_gnt_comparison(figure_path, averaged.base_times,
                                predicted_values, actual_values, args)
    summary = {
        "method": ("train on multiple generated averaged sets; validate on current averaged sample"
                   if is_validation else
                   "average eight runs first; train and fit the same averaged windows"),
        "source_runs": list(args.runs),
        "averaging_order": "window -> substep -> link -> mean across runs",
        "observed_steps": int(averaged.observed.shape[1]),
        "prediction_steps": int(averaged.target.shape[1]),
        "observed_ratio": float(averaged.observed.shape[1] / averaged.complete.shape[1]),
        "training_and_prediction_samples_are_the_same": not is_validation,
        "evaluation_role": "validation" if is_validation else "training_fit",
        "temperature_target": TEMPERATURE_TARGET,
        "results": [{"test_run":"mean_of_8",
                     "training_runs":([d.run for d in training_sets]
                                      if is_validation else list(args.runs)),
                     "training_info":info,"training_sample_count":sample_count,
                     "metrics":metrics,"prediction_npz":str(npz_path),"prediction_yaml":str(yaml_path)}],
        "gnt_comparison": comparison,
        "summary_figure": str(figure_path),
    }
    with (output_dir / args.summary_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(f"{'独立验证' if is_validation else '平均后训练'}完成："
          f"未来均值MAE={metrics['future_mean']['MAE']:.6f}；"
          f"窗口GNT MAE={metrics['window_gnt']['MAE']:.3f}")
    print(f"输出目录：{output_dir}")


def residual_features(observed: np.ndarray):
    """Build percentage-change features using only current observations, without global normalization.
        Use a minimum scale of 1 for low traffic to avoid near-zero division; the input determines the scale."""
    observed = np.asarray(observed, dtype=np.float64)
    scale = np.maximum(observed.mean(axis=1), 1.0)
    last = observed[:, -1, :]
    features = 100.0 * (observed - last[:, None, :]) / scale[:, None, :]
    return features.astype(np.float32), scale, last


def build_lstm_model(tf, observed_steps, hidden_size, dropout, learning_rate):
    """Learn residuals relative to the last observation; zero-initialized outputs start as last-value persistence."""
    inputs = tf.keras.Input(shape=(observed_steps, 1), name="observed_relative_change")
    encoded = tf.keras.layers.LSTM(hidden_size, dropout=dropout)(inputs)
    outputs = tf.keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros",
        name="future_mean_residual_percent",
    )(encoded)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse", metrics=["mae"],
    )
    return model


def build_future_mean_samples(datasets, input_steps, prediction_steps):
    x_samples, y_samples = [], []
    for dataset in datasets:
        if (dataset.observed.shape[1] != input_steps
                or dataset.target.shape[1] != prediction_steps):
            raise ValueError(f"第{dataset.run}次实验的小段数量不一致。")
        features, scale, last = residual_features(dataset.observed)
        future_mean = dataset.target.mean(axis=1, dtype=np.float64)
        active = np.any(dataset.observed > 0, axis=1)
        # This pipeline perturbs existing baseline traffic; zero baselines remain zero throughout the window.
        # Real logs with newly active links require a separate activation model; never silently discard them.
        if np.any((~active) & (future_mean > 0)):
            raise ValueError("发现观测全零但未来激活的链路；当前合成数据模型不支持该任务。")
        x_samples.append(features.transpose(0, 2, 1).reshape(-1, input_steps, 1)[active.ravel()])
        residual = 100.0 * (future_mean - last) / scale
        y_samples.append(residual.reshape(-1, 1)[active.ravel()])
    x = np.concatenate(x_samples).astype(np.float32)
    y = np.concatenate(y_samples).astype(np.float32)
    if len(x) == 0:
        raise ValueError("数据中没有非零链路样本。")
    return x, y


def predict_future_mean(model, observed, prediction_steps, batch_size):
    """Return a legacy-NPZ-compatible shape; repeated means are a storage convention, not segment-level forecasts."""
    features, scale, last = residual_features(observed)
    n, steps, links = observed.shape
    output = model.predict(
        features.transpose(0, 2, 1).reshape(-1, steps, 1),
        batch_size=batch_size, verbose=0,
    ).reshape(n, links)
    if not np.all(np.isfinite(output)):
        raise ValueError("模型产生了非有限预测，请检查训练损失。")
    predicted_mean = np.maximum(last + output * scale / 100.0, 0.0)
    predicted_mean[np.all(observed == 0, axis=1)] = 0.0
    return np.repeat(predicted_mean[:, None, :], prediction_steps, axis=1).astype(np.float32)


def split_training_runs(training_runs, validation_ratio, seed):
    """Choose whole validation runs only from the outer training split; never inspect outer test labels."""
    if len(training_runs) < 2:
        raise ValueError("分组验证至少需要2个训练实验，即总计至少3个实验。")
    indices = np.random.default_rng(seed).permutation(len(training_runs))
    count = min(len(training_runs)-1, max(1, round(len(training_runs)*validation_ratio)))
    val_indices = set(indices[:count].tolist())
    fit_runs = [d for i, d in enumerate(training_runs) if i not in val_indices]
    validation_runs = [d for i, d in enumerate(training_runs) if i in val_indices]
    return fit_runs, validation_runs


def set_random_seed(tf, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train_and_predict(
    tf,
    training_runs: Sequence[RunData],
    test_run: RunData,
    args: argparse.Namespace,
):
    input_steps = test_run.observed.shape[1]
    prediction_steps = test_run.target.shape[1]
    fold_seed = args.seed + int(test_run.run)
    set_random_seed(tf, fold_seed)
    fit_runs, validation_runs = split_training_runs(
        training_runs, args.validation_ratio, fold_seed,
    )
    x_train, y_train = build_future_mean_samples(fit_runs, input_steps, prediction_steps)
    x_validation, y_validation = build_future_mean_samples(validation_runs, input_steps, prediction_steps)
    print(f"内部拟合组={[d.run for d in fit_runs]}，验证组={[d.run for d in validation_runs]}")

    model = build_lstm_model(
        tf=tf,
        observed_steps=x_train.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(args.patience // 3, 3),
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_validation, y_validation),
        shuffle=True,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    prediction = predict_future_mean(
        model=model,
        observed=test_run.observed,
        prediction_steps=prediction_steps,
        batch_size=args.batch_size,
    )
    return (
        model,
        history.history,
        prediction,
        len(x_train),
        {
            "model_kind": "last_value_residual_lstm",
            "fit_runs": [d.run for d in fit_runs],
            "validation_runs": [d.run for d in validation_runs],
            "validation_sample_count": len(x_validation),
            "normalization": "observed_only_percent_change; scale=max(observed_mean,1)",
            "future_expansion": "constant_mean; not a per-step forecast",
            "seed": fold_seed,
        },
    )


def train_all_runs_in_sample(tf, datasets: Sequence[RunData], args: argparse.Namespace):
    """Train one model on all eight runs for subsequent in-sample fitting on those same runs only."""
    input_steps = datasets[0].observed.shape[1]
    prediction_steps = datasets[0].target.shape[1]
    set_random_seed(tf, args.seed)
    x_train, y_train = build_future_mean_samples(datasets, input_steps, prediction_steps)
    model = build_lstm_model(
        tf=tf,
        observed_steps=input_steps,
        hidden_size=args.window_in_sample_hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_train, y_train),
        epochs=args.window_in_sample_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.window_in_sample_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=.5,
                patience=max(3, args.window_in_sample_patience // 3), min_lr=1e-5,
            ),
        ],
        verbose=args.verbose,
    )
    info = {
        "model_kind": "all-runs training-set fit; last-value residual LSTM",
        "fit_runs": [d.run for d in datasets],
        "validation_runs": [d.run for d in datasets],
        "prediction_runs": [d.run for d in datasets],
        "training_and_prediction_samples_are_the_same": True,
        "observed_steps": input_steps,
        "prediction_steps": prediction_steps,
        "observed_ratio": input_steps / (input_steps + prediction_steps),
        "training_sample_count": len(x_train),
        "epochs_ran": len(history.history["loss"]),
    }
    return model, history.history, len(x_train), info


def train_generated_sets(tf, training_sets, validation_set, args):
    """Fit on generated seeds and select checkpoints on the untouched current sample."""
    input_steps=validation_set.observed.shape[1]
    prediction_steps=validation_set.target.shape[1]
    set_random_seed(tf,args.seed)
    x_train,y_train=build_future_mean_samples(training_sets,input_steps,prediction_steps)
    x_val,y_val=build_future_mean_samples([validation_set],input_steps,prediction_steps)
    model=build_lstm_model(tf,input_steps,args.window_in_sample_hidden_size,
                           args.dropout,args.learning_rate)
    history=model.fit(
        x_train,y_train,validation_data=(x_val,y_val),
        epochs=args.window_in_sample_epochs,batch_size=args.batch_size,shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=args.window_in_sample_patience,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",factor=.5,
                patience=max(3,args.window_in_sample_patience//3),min_lr=1e-5)],
        verbose=args.verbose)
    info={"model_kind":"multi-seed synthetic training; current averaged sample validation",
          "fit_sets":[d.run for d in training_sets],"validation_set":validation_set.run,
          "training_sample_count":len(x_train),"validation_sample_count":len(x_val),
          "training_and_validation_are_same":False,
          "shared_baseline_warning":"generated training and validation share window baseline flows",
          "epochs_ran":len(history.history["loss"]),
          "best_epoch":int(np.argmin(history.history["val_loss"])+1)}
    return model,history.history,len(x_train),info


def compute_entropy(flow_values: np.ndarray) -> float:
    values = np.maximum(np.asarray(flow_values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probability = values / total
    positive = probability > 0.0
    return float(
        -(probability[positive] * np.log(probability[positive])).sum()
    )


def compute_domain_state(
    window_link_flow: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    domain_nodes: Sequence[int],
) -> Tuple[float, float]:
    domain_set = {str(node) for node in domain_nodes}
    mask = np.asarray(
        [
            source in domain_set and target in domain_set
            for source, target in zip(edge_sources, edge_targets)
        ],
        dtype=bool,
    )
    domain_flow = np.asarray(window_link_flow[mask], dtype=np.float64)
    total_flow = float(domain_flow.sum())
    entropy = compute_entropy(domain_flow)
    return total_flow, entropy


def clip_gnt(value: float) -> float:
    return float(np.clip(value, GNT_CLIP_MIN, GNT_CLIP_MAX))


def average_domain_temperature(domains: Sequence[dict]) -> float:
    """Evaluation convention: clip each domain before averaging; replace nulls with zero and report their count separately."""
    values = {int(d["domain_id"]): d.get("T") for d in domains}
    temperatures = [values.get(i) for i in range(len(DOMAINS))]
    if any(t is not None and not np.isfinite(t) for t in temperatures):
        raise ValueError("域温度包含非有限数值。")
    return float(np.mean([0.0 if t is None else clip_gnt(t) for t in temperatures]))


def compute_gnt_records(
    base_times: np.ndarray,
    window_link_flow: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
) -> List[dict]:
    """Compute GNT from mean traffic in adjacent 100-unit windows.
        The first window has no predecessor, so all domain temperatures and mean GNT are zero."""
    previous_states: List[Tuple[float, float]] | None = None
    records: List[dict] = []
    if len(base_times) != len(window_link_flow):
        raise ValueError("温度时间轴与流量窗口数量不一致。")
    if len(base_times) > 1 and not np.allclose(np.diff(base_times), 100.0):
        raise ValueError("温度必须由连续的100单位窗口差分计算。")

    for time_value, link_flow in zip(base_times, window_link_flow):
        current_states = [
            compute_domain_state(
                window_link_flow=link_flow,
                edge_sources=edge_sources,
                edge_targets=edge_targets,
                domain_nodes=domain,
            )
            for domain in DOMAINS
        ]

        domain_temperatures = []
        average_values = []
        for domain_id, (total_flow, entropy) in enumerate(current_states):
            if previous_states is None:
                delta_flow = 0.0
                delta_entropy = 0.0
                temperature = None
                average_value = 0.0
            else:
                previous_flow, previous_entropy = previous_states[domain_id]
                delta_flow = total_flow - previous_flow
                delta_entropy = entropy - previous_entropy
                if abs(delta_entropy) < ENTROPY_EPS:
                    temperature = None
                    average_value = 0.0
                else:
                    # Preserve raw GNT during generation; do not clip here.
                    temperature = delta_flow / delta_entropy
                    average_value = temperature

            domain_temperatures.append(
                {
                    "domain_id": domain_id,
                    "F": total_flow,
                    "H": entropy,
                    "delta_F": delta_flow,
                    "delta_H": delta_entropy,
                    "T": temperature,
                }
            )
            average_values.append(average_value)

        # Save raw mean GNT and handle extreme values only during plotting.
        average_temperature = float(np.mean(average_values))
        records.append(
            {
                "time": float(time_value),
                "domain_temperatures": domain_temperatures,
                "average_network_temperature": average_temperature,
                "comparison_average_network_temperature": average_domain_temperature(domain_temperatures),
                "valid_domain_count": sum(d["T"] is not None for d in domain_temperatures),
            }
        )
        previous_states = current_states

    return records


def calculate_metrics(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> dict:
    difference = prediction.astype(np.float64) - actual.astype(np.float64)
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    nonzero = actual > 1e-8
    if np.any(nonzero):
        mape = float(
            np.mean(np.abs(difference[nonzero]) / actual[nonzero]) * 100.0
        )
    else:
        mape = 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}


def error_metrics(prediction: np.ndarray, actual: np.ndarray) -> dict:
    """Do not report MAPE because temperatures may be negative or close to zero."""
    prediction, actual = np.asarray(prediction, float), np.asarray(actual, float)
    if prediction.shape != actual.shape or prediction.size == 0:
        raise ValueError("指标需要同形状且非空的预测/真实配对。")
    if not (np.all(np.isfinite(prediction)) and np.all(np.isfinite(actual))):
        raise ValueError("指标配对包含缺失或非有限值。")
    diff = prediction - actual
    return {"sample_count": int(diff.size), "MAE": float(np.mean(abs(diff))),
            "RMSE": float(np.sqrt(np.mean(diff**2)))}


def evaluate_prediction(dataset: RunData, predicted_target: np.ndarray, start_time=500.0):
    """Evaluate all methods against the same complete-traffic labels; use the first window only for warmup."""
    if (predicted_target.shape != dataset.target.shape
            or not np.all(np.isfinite(predicted_target)) or np.any(predicted_target < 0)):
        raise ValueError("预测数组必须与目标同形状、非负且有限。")
    mask = dataset.base_times >= start_time
    mask[0] = False
    if not np.any(mask):
        raise ValueError("评估区间内没有非预热窗口。")
    actual_flow = dataset.complete.mean(axis=1, dtype=np.float64)
    predicted_flow = (dataset.observed.sum(axis=1, dtype=np.float64)
                      + predicted_target.sum(axis=1, dtype=np.float64)) / dataset.complete.shape[1]
    def records(flow):
        return compute_gnt_records(dataset.base_times, flow, dataset.edge_sources, dataset.edge_targets)
    actual_gnt, predicted_gnt = records(actual_flow), records(predicted_flow)
    actual_mean = dataset.target.mean(axis=1, dtype=np.float64)
    metrics = {
        "future_mean": calculate_metrics(predicted_target.mean(axis=1, dtype=np.float64)[mask], actual_mean[mask]),
        "complete_window_mean": calculate_metrics(predicted_flow[mask], actual_flow[mask]),
        "window_gnt": error_metrics(gnt_values(predicted_gnt)[mask], gnt_values(actual_gnt)[mask]),
    }
    actual_domains = [r["domain_temperatures"] for r, use in zip(actual_gnt, mask) if use]
    predicted_domains = [r["domain_temperatures"] for r, use in zip(predicted_gnt, mask) if use]
    metrics["stability"] = {
        "domain_count": len(actual_domains) * len(DOMAINS),
        "actual_undefined": sum(d["T"] is None for r in actual_domains for d in r),
        "predicted_undefined": sum(d["T"] is None for r in predicted_domains for d in r),
        "actual_abs_delta_H_lt_0_01": sum(abs(d["delta_H"]) < 0.01 for r in actual_domains for d in r),
        "delta_H_sign_flips": sum(a["delta_H"]*p["delta_H"] < 0
            for ar, pr in zip(actual_domains, predicted_domains) for a, p in zip(ar, pr)),
    }
    return metrics, actual_flow, predicted_flow, actual_gnt, predicted_gnt


def baseline_predictions(observed: np.ndarray, prediction_steps: int) -> dict:
    return {name: np.repeat(mean[:, None, :], prediction_steps, axis=1)
            for name, mean in {"last_value": observed[:, -1, :],
                               "prefix_mean": observed.mean(axis=1, dtype=np.float64)}.items()}


def load_saved_prediction(path: Path, dataset: RunData) -> np.ndarray:
    """Reevaluate legacy LSTM predictions from existing NPZ files; reject predictions inconsistent with current inputs."""
    with np.load(path) as saved:
        for name, expected in (("base_times", dataset.base_times),
                               ("edge_sources", dataset.edge_sources),
                               ("edge_targets", dataset.edge_targets),
                               ("observed_link_flow", dataset.observed),
                               ("actual_complete_link_flow", dataset.complete)):
            actual = saved[name].astype(str) if name.startswith("edge_") else saved[name]
            if not np.array_equal(actual, expected):
                raise ValueError(f"{path} 的{name}与当前数据不一致，不能复用。")
        predicted = saved["predicted_target_link_flow"].copy()
    if (predicted.shape != dataset.target.shape or not np.all(np.isfinite(predicted))
            or np.any(predicted < 0)):
        raise ValueError(f"{path} 的预测形状或数值不合法。")
    return predicted


def build_flow_list(
    link_flow: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
) -> List[dict]:
    return [
        {
            "source": parse_node(source),
            "target": parse_node(target),
            "flow": float(flow),
        }
        for source, target, flow in zip(
            edge_sources,
            edge_targets,
            link_flow,
        )
        if flow > 0.0
    ]


def save_prediction_npz(
    output_path: Path,
    dataset: RunData,
    predicted_target: np.ndarray,
    reconstructed: np.ndarray,
    predicted_window_flow: np.ndarray,
    actual_window_flow: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        base_times=dataset.base_times,
        edge_sources=dataset.edge_sources,
        edge_targets=dataset.edge_targets,
        observed_link_flow=dataset.observed,
        predicted_target_link_flow=predicted_target,
        actual_target_link_flow=dataset.target,
        reconstructed_link_flow=reconstructed,
        actual_complete_link_flow=dataset.complete,
        predicted_window_link_flow=predicted_window_flow,
        actual_window_link_flow=actual_window_flow,
    )


def save_prediction_yaml(
    output_path: Path,
    dataset: RunData,
    predicted_window_flow: np.ndarray,
    actual_window_flow: np.ndarray,
    predicted_gnt: Sequence[dict],
    actual_gnt: Sequence[dict],
    metrics: dict,
    training_runs: Sequence[str],
    training_sample_count: int,
    training_info: dict,
) -> None:
    records = []
    for index, time_value in enumerate(dataset.base_times):
        records.append(
            {
                "time": float(time_value),
                "predicted_global_link_flow": build_flow_list(
                    predicted_window_flow[index],
                    dataset.edge_sources,
                    dataset.edge_targets,
                ),
                "predicted_domain_temperatures": predicted_gnt[index][
                    "domain_temperatures"
                ],
                "predicted_average_network_temperature": predicted_gnt[index][
                    "average_network_temperature"
                ],
                "actual_average_network_temperature": actual_gnt[index][
                    "average_network_temperature"
                ],
                "actual_domain_temperatures": actual_gnt[index]["domain_temperatures"],
                "predicted_comparison_average_network_temperature": average_domain_temperature(predicted_gnt[index]["domain_temperatures"]),
                "actual_comparison_average_network_temperature": average_domain_temperature(actual_gnt[index]["domain_temperatures"]),
            }
        )

    content = {
        "description": (
            "LSTM prediction for each 100-time-unit window and GNT results"
        ),
        "test_run": dataset.run,
        "training_runs": list(training_runs),
        "training_sample_count": training_sample_count,
        "training_info": training_info,
        "temperature_target": TEMPERATURE_TARGET,
        "observed_steps": int(dataset.observed.shape[1]),
        "prediction_steps": int(dataset.target.shape[1]),
        "observed_ratio": float(
            dataset.observed.shape[1] / dataset.complete.shape[1]
        ),
        "prediction_rule": (
            "single-link future-mean prediction: use "
            f"{dataset.observed.shape[1]} observed steps to directly predict "
            f"the mean flow of the next {dataset.target.shape[1]} steps"
        ),
        "window_flow_rule": f"mean of all {dataset.complete.shape[1]} substeps",
        "temperature_rule": "T = delta_F / delta_H between adjacent windows",
        "undefined_temperature_in_average": 0,
        "temperature_clip": "per_domain_before_mean_for_comparison; raw_values_retained",
        "entropy_epsilon": ENTROPY_EPS,
        "data_scope": "synthetic substeps; window GNT, not recorded event GNT",
        "prediction_metrics": metrics,
        "records": records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def save_training_history(
    output_path: Path,
    history: dict,
) -> None:
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, allow_unicode=True, sort_keys=False)


def gnt_values(records: Sequence[dict]) -> np.ndarray:
    return np.asarray(
        [average_domain_temperature(record["domain_temperatures"]) for record in records],
        dtype=np.float64,
    )


def configure_gnt_axis(
    axis,
    base_times: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Apply a shared 100-unit time axis and publication plot styling."""
    axis.set_xlabel("Time")
    axis.set_ylabel("Average Network Temperature")
    axis.set_xticks(base_times)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, linestyle="--", alpha=0.45)
    if args.gnt_y_min is not None or args.gnt_y_max is not None:
        bottom, top = axis.get_ylim()
        axis.set_ylim(
            args.gnt_y_min if args.gnt_y_min is not None else bottom,
            args.gnt_y_max if args.gnt_y_max is not None else top,
        )


def plot_run_gnt_comparison(
    output_path: Path,
    run: str,
    base_times: np.ndarray,
    predicted_gnt: Sequence[dict],
    actual_gnt: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    """Plot true and predicted mean GNT for each 100-unit window of one run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (base_times >= args.evaluation_start)
    mask[0] = False
    predicted = gnt_values(predicted_gnt)[mask]
    actual = gnt_values(actual_gnt)[mask]
    base_times = base_times[mask]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.vlines(base_times, actual, predicted, color="#8c8c8c", alpha=.45,
                linewidth=1.0, label="Prediction residual")
    axis.plot(
        base_times,
        actual,
        marker="o",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
        label="Actual window GNT",
    )
    axis.plot(
        base_times,
        predicted,
        marker="^",
        markersize=6,
        linewidth=2,
        linestyle="--",
        markerfacecolor="none",
        markeredgewidth=1.5,
        zorder=4,
        label="Predicted window GNT",
    )
    mae=float(np.mean(np.abs(predicted-actual)))
    role = "Validation" if args.window_evaluation == "augmented_validation" else "Training-fit"
    axis.text(.02,.03,f"{role} MAE = {mae:.1f}",transform=axis.transAxes,
              fontsize=9,color="#444444")
    scope = "validation" if args.window_evaluation == "augmented_validation" else "training-set fit"
    axis.set_title(f"Window GNT {scope} ({run}); connectors show residuals")
    configure_gnt_axis(axis, base_times, args)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def confidence_interval_95(values: np.ndarray) -> np.ndarray:
    """Compute the 95% confidence-interval half-width for the across-run mean."""
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(
        values.shape[0]
    )
    return 1.96 * standard_error


def plot_summary_gnt_comparison(
    output_path: Path,
    base_times: np.ndarray,
    predicted_values: np.ndarray,
    actual_values: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Plot mean GNT across eight runs with 95% confidence intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (base_times >= args.evaluation_start)
    mask[0] = False
    predicted_values, actual_values = predicted_values[:, mask], actual_values[:, mask]
    base_times = base_times[mask]
    predicted_mean = np.mean(predicted_values, axis=0)
    actual_mean = np.mean(actual_values, axis=0)
    predicted_ci = confidence_interval_95(predicted_values)
    actual_ci = confidence_interval_95(actual_values)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.vlines(base_times, actual_mean, predicted_mean, color="#8c8c8c", alpha=.45,
                linewidth=1.0, label="Prediction residual")
    axis.plot(
        base_times,
        actual_mean,
        marker="o",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
        label="Actual window GNT",
    )
    axis.fill_between(
        base_times,
        actual_mean - actual_ci,
        actual_mean + actual_ci,
        alpha=0.18,
    )
    axis.plot(
        base_times,
        predicted_mean,
        marker="^",
        markersize=6,
        linewidth=2,
        linestyle="--",
        markerfacecolor="none",
        markeredgewidth=1.5,
        zorder=4,
        label="Predicted window GNT",
    )
    axis.fill_between(
        base_times,
        predicted_mean - predicted_ci,
        predicted_mean + predicted_ci,
        alpha=0.18,
    )
    mae=float(np.mean(np.abs(predicted_mean-actual_mean)))
    rmse=float(np.sqrt(np.mean((predicted_mean-actual_mean)**2)))
    role = "Validation" if args.window_evaluation == "augmented_validation" else "Training-fit"
    axis.text(.02,.03,f"{role} MAE = {mae:.1f}; RMSE = {rmse:.1f}",
              transform=axis.transAxes,fontsize=9,color="#444444")
    scope = "validation" if args.window_evaluation == "augmented_validation" else "training-set fit"
    axis.set_title(f"Window GNT {scope}; connectors show residuals")
    configure_gnt_axis(axis, base_times, args)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "默认直接预测原始日志GNT；window模式保留合成窗口流量实验。"
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent / "GNT")
    parser.add_argument("--target-mode", choices=["mean", "recorded", "window"], default="window",
                        help="默认使用全部8次实验训练前35%窗口模型。")
    parser.add_argument("--window-evaluation",
                        choices=["augmented_validation", "in_sample", "loro"],
                        default="augmented_validation",
                        help="默认用多组生成数据训练、当前8次平均样本验证。")
    parser.add_argument("--generated-training-sets", type=int, default=8,
                        help="augmented_validation模式生成的平均训练集数量。")
    parser.add_argument("--generated-training-dir", type=Path, default=None,
                        help="生成训练集目录；默认位于本次输出目录下。")
    parser.add_argument("--window-in-sample-epochs", type=int, default=10)
    parser.add_argument("--window-in-sample-hidden-size", type=int, default=64)
    parser.add_argument("--window-in-sample-patience", type=int, default=20)
    parser.add_argument("--raw-gnt-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "result_storage" / "EO")
    parser.add_argument("--recorded-history", type=int, default=3,
                        help="recorded模式使用的历史窗口数量。")
    parser.add_argument("--recorded-end", type=float, default=2000.0)
    parser.add_argument("--mean-start", type=float, default=500.0)
    parser.add_argument("--mean-train-end", type=float, default=1400.0)
    parser.add_argument("--mean-validation-end", type=float, default=1700.0)
    parser.add_argument("--mean-evaluation", choices=["in_sample", "chronological"], default="in_sample",
                        help="默认在8实验平均后的同一训练集上拟合和预测；chronological保留时间外推。")
    parser.add_argument("--mean-epochs", type=int, default=800)
    parser.add_argument("--mean-hidden-size", type=int, default=64)
    parser.add_argument("--mean-patience", type=int, default=80)
    parser.add_argument("--min-residual-improvement", type=float, default=0.02,
                        help="recorded模式中采用LSTM残差所需的最小独立验证相对改善，默认2%。")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录；训练默认写回base-dir，重新评估默认写到base-dir/evaluation。")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="不训练，读取已有预测NPZ，统一温度口径并计算基线。")
    parser.add_argument("--evaluation-start", type=float, default=500.0,
                        help="指标和绘图起点；第一个数据窗口始终仅作预热。")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[str(run) for run in range(1, 9)],
    )
    parser.add_argument(
        "--input-name",
        default="lstm_traffic_500_2000.npz",
    )
    parser.add_argument(
        "--prediction-npz-name",
        default="lstm_prediction_500_2000.npz",
    )
    parser.add_argument(
        "--prediction-yaml-name",
        default="lstm_prediction_gnt_500_2000.yaml",
    )
    parser.add_argument(
        "--history-name",
        default="lstm_training_history.yaml",
    )
    parser.add_argument(
        "--summary-name",
        default="lstm_gnt_prediction_summary.yaml",
    )
    parser.add_argument(
        "--run-figure-name",
        default="lstm_gnt_prediction_comparison.png",
    )
    parser.add_argument(
        "--summary-figure-name",
        default="lstm_gnt_prediction_mean_comparison.png",
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--gnt-y-min",
        type=float,
        default=None,
        help="可选：设置GNT图纵轴下限。",
    )
    parser.add_argument(
        "--gnt-y-max",
        type=float,
        default=None,
        help="可选：设置GNT图纵轴上限。",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="在每个实验目录保存训练后的.keras模型。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.runs) < (2 if args.evaluate_only else 3):
        raise ValueError("训练需要至少3组数据（拟合、验证、测试）；重新评估至少2组。")
    if len(set(args.runs)) != len(args.runs):
        raise ValueError("runs中不能包含重复实验编号。")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs和batch-size必须大于0。")
    if args.hidden_size <= 0:
        raise ValueError("hidden-size必须大于0。")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate) or args.patience <= 0:
        raise ValueError("learning-rate须为正有限数，patience须大于0。")
    if not np.isfinite(args.evaluation_start):
        raise ValueError("evaluation-start必须为有限数。")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout必须位于[0,1)内。")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio必须位于(0,1)内。")
    if args.figure_dpi <= 0:
        raise ValueError("figure-dpi必须大于0。")
    if args.recorded_history < 1 or not np.isfinite(args.recorded_end):
        raise ValueError("recorded-history和recorded-end无效。")
    if min(args.mean_epochs,args.mean_hidden_size,args.mean_patience) < 1:
        raise ValueError("mean模式训练参数必须为正数。")
    if min(args.window_in_sample_epochs,args.window_in_sample_hidden_size,
           args.window_in_sample_patience) < 1:
        raise ValueError("window训练集拟合参数必须为正数。")
    if args.generated_training_sets < 2:
        raise ValueError("generated-training-sets至少为2。")
    if not (np.isfinite(args.mean_start) and np.isfinite(args.mean_train_end)
            and np.isfinite(args.mean_validation_end)
            and args.mean_start <= args.mean_train_end < args.mean_validation_end < args.recorded_end):
        raise ValueError("mean模式时间划分必须满足start <= train-end < validation-end < recorded-end。")
    if not 0 <= args.min_residual_improvement < 1:
        raise ValueError("min-residual-improvement必须位于[0,1)内。")
    if (
        args.gnt_y_min is not None
        and args.gnt_y_max is not None
        and args.gnt_y_max <= args.gnt_y_min
    ):
        raise ValueError("gnt-y-max必须大于gnt-y-min。")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.target_mode == "mean":
        if args.evaluate_only:
            raise ValueError("mean模式结果可直接由picture.py读取，无需--evaluate-only。")
        import mean_lstm
        mean_lstm.run(args)
        return
    if args.target_mode == "recorded":
        if args.evaluate_only:
            raise ValueError("recorded模式直接读取已保存结果即可，无需--evaluate-only。")
        import recorded_lstm
        recorded_lstm.run(args)
        return
    tf = None if args.evaluate_only else require_tensorflow()
    output_dir = args.output_dir or (args.base_dir / "evaluation" if args.evaluate_only else
                                    args.base_dir / "window_mean_in_sample" if args.window_evaluation == "in_sample" else
                                    args.base_dir / "window_generated_validation" if args.window_evaluation == "augmented_validation" else
                                    args.base_dir)
    datasets = load_all_runs(args.base_dir, args.runs, args.input_name)
    if not args.evaluate_only and args.window_evaluation == "augmented_validation":
        import multi_training_generator
        import window_recorded_target
        generated_dir = args.generated_training_dir or output_dir / "generated_training_sets"
        training_sets = multi_training_generator.generate(
            datasets=datasets,
            runs=args.runs,
            count=args.generated_training_sets,
            seed=args.seed + 100000,
            output_dir=generated_dir,
        )
        print(f"已生成{len(training_sets)}组训练集：{generated_dir}")
        print("当前8次实验的平均流量样本未加入训练，只用于recorded GNT验证。")
        averaged = average_run_data(datasets, args.runs)
        window_recorded_target.run(tf, training_sets, averaged, args, output_dir)
        return
    if not args.evaluate_only and args.window_evaluation == "in_sample":
        run_averaged_window_fit(tf, datasets, args, output_dir)
        return
    reference_times = datasets[args.runs[0]].base_times
    mask = reference_times >= args.evaluation_start
    mask[0] = False
    if not np.any(mask):
        raise ValueError("评估范围内没有非预热窗口。")
    summary = {
        "method": ("saved_predictions_re_evaluation" if args.evaluate_only else
                   "all eight runs: train and predict same samples" if args.window_evaluation == "in_sample" else
                   "leave-one-run-out; grouped inner validation"),
        "temperature_target": TEMPERATURE_TARGET,
        "temperature_clip": "per_domain_before_mean",
        "entropy_epsilon": ENTROPY_EPS,
        "evaluation_times": reference_times[mask].astype(float).tolist(),
        "runs": list(args.runs), "domains": DOMAINS, "results": [],
    }
    all_actual_gnt = []
    all_method_gnt = {name: [] for name in ("lstm", "last_value", "prefix_mean")}
    print("比较目标：相邻窗口平均流量计算的GNT；输入每窗口前7/20段，预测后13/20段。")
    shared = None
    if not args.evaluate_only and args.window_evaluation == "in_sample":
        shared = train_all_runs_in_sample(tf, [datasets[r] for r in args.runs], args)
    for test_name in args.runs:
        test_run = datasets[test_name]
        training_names = [r for r in args.runs if r != test_name]
        training_runs = [datasets[r] for r in training_names]
        if args.window_evaluation == "in_sample" and not args.evaluate_only:
            print(f"\n[Run {test_name}] 训练组={args.runs}，同训练集预测组={test_name}")
        else:
            print(f"\n[Run {test_name}] 外层训练候选组={training_names}，测试组={test_name}")
        if args.evaluate_only:
            source_path = args.base_dir / test_name / args.prediction_npz_name
            predicted_target = load_saved_prediction(source_path, test_run)
            model, history, sample_count = None, {}, 0
            info = {"model_kind": "saved_prediction; training_not_repeated", "source_npz": str(source_path)}
            # Historical training splits are unknown; do not attribute old predictions to the current configuration.
            recorded_training_names = []
        elif args.window_evaluation == "in_sample":
            model, shared_history, sample_count, shared_info = shared
            history = shared_history
            predicted_target = predict_future_mean(
                model, test_run.observed, test_run.target.shape[1], args.batch_size,
            )
            info = {**shared_info, "current_prediction_run": test_name}
            recorded_training_names = list(args.runs)
        else:
            model, history, predicted_target, sample_count, info = train_and_predict(tf, training_runs, test_run, args)
            recorded_training_names = training_names
        metrics, actual_flow, predicted_flow, actual_gnt, predicted_gnt = evaluate_prediction(
            test_run, predicted_target, args.evaluation_start,
        )
        baselines = {}
        all_actual_gnt.append(gnt_values(actual_gnt))
        all_method_gnt["lstm"].append(gnt_values(predicted_gnt))
        for name, forecast in baseline_predictions(test_run.observed, test_run.target.shape[1]).items():
            result = evaluate_prediction(test_run, forecast, args.evaluation_start)
            baselines[name] = result[0]
            all_method_gnt[name].append(gnt_values(result[4]))
        metrics["baselines"] = baselines
        run_dir = output_dir / test_name
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path, yaml_path = run_dir / args.prediction_npz_name, run_dir / args.prediction_yaml_name
        save_prediction_npz(npz_path, test_run, predicted_target,
                            np.concatenate([test_run.observed, predicted_target], axis=1), predicted_flow, actual_flow)
        save_prediction_yaml(yaml_path, test_run, predicted_flow, actual_flow,
                             predicted_gnt, actual_gnt, metrics, recorded_training_names, sample_count, info)
        if not args.evaluate_only:
            save_training_history(run_dir / args.history_name, history)
            if args.save_models:
                model.save(run_dir / "lstm_traffic_model.keras")
        plot_run_gnt_comparison(run_dir / args.run_figure_name, test_name, reference_times,
                                predicted_gnt, actual_gnt, args)
        summary["results"].append({
            "test_run": test_name, "training_runs": recorded_training_names,
            "training_info": info, "training_sample_count": sample_count,
            "metrics": metrics, "prediction_npz": str(npz_path), "prediction_yaml": str(yaml_path),
        })
        print(f"[Run {test_name}] 未来均值MAE={metrics['future_mean']['MAE']:.6f}；"
              f"窗口GNT MAE={metrics['window_gnt']['MAE']:.3f}；"
              f"末值基线GNT MAE={baselines['last_value']['window_gnt']['MAE']:.3f}")
        if tf is not None and args.window_evaluation != "in_sample":
            tf.keras.backend.clear_session()
    actual_values = np.asarray(all_actual_gnt)
    summary["gnt_comparison"] = {}
    for name, forecasts in all_method_gnt.items():
        values = np.asarray(forecasts)
        summary["gnt_comparison"][name] = {
            "per_run_time": error_metrics(values[:, mask], actual_values[:, mask]),
            "mean_curve": error_metrics(values[:, mask].mean(axis=0), actual_values[:, mask].mean(axis=0)),
        }
    figure_path = output_dir / args.summary_figure_name
    plot_summary_gnt_comparison(figure_path, reference_times, np.asarray(all_method_gnt["lstm"]), actual_values, args)
    summary["summary_figure"] = str(figure_path)
    with (output_dir / args.summary_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(f"\n完成 {len(args.runs)} 组，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
