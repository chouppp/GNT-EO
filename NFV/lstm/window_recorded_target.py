"""Use the first 35% window flow to predict the recorded GNT target directly."""
from pathlib import Path
import numpy as np
import yaml
import gnt_dataset as recorded


DOMAINS = [
    [7, 2, 4, 5, 6], [18, 1, 3, 8, 10], [20, 9, 15, 17, 19],
    [12, 0, 21, 22, 23], [11, 14, 24, 13, 16],
]


def observed_domain_features(data):
    """Convert seven link-flow snapshots to five-domain F/H sequences."""
    flow = np.asarray(data.observed, dtype=np.float64)
    source = np.asarray(data.edge_sources).astype(str)
    target = np.asarray(data.edge_targets).astype(str)
    features = []
    for nodes in DOMAINS:
        members = {str(node) for node in nodes}
        mask = np.array([s in members and t in members for s, t in zip(source, target)])
        domain_flow = flow[:, :, mask]
        total = domain_flow.sum(axis=2)
        probability = np.divide(
            domain_flow, total[:, :, None],
            out=np.zeros_like(domain_flow), where=total[:, :, None] > 0,
        )
        entropy = -(np.where(probability > 0, probability * np.log(
            np.maximum(probability, 1e-12)), 0.0)).sum(axis=2)
        features.extend((np.log1p(total), entropy))
    return np.stack(features, axis=2).astype(np.float32)


def recorded_mean_targets(raw_dir, runs, times):
    per_run = []
    sources = {}
    for run in runs:
        path = Path(raw_dir) / run / "resource_usage.yaml"
        recorded_times, states, digest = recorded.read_states(path)
        mapping = {float(t): state[2] for t, state in zip(recorded_times, states)}
        missing = [float(t) for t in times if float(t) not in mapping]
        if missing:
            raise ValueError(f"{path}缺少真实GNT时间点：{missing}")
        per_run.append([mapping[float(t)].mean() * recorded.LIMIT for t in times])
        sources[run] = {"path": str(path.resolve()), "sha256": digest}
    return np.mean(per_run, axis=0).astype(np.float32), sources


def build_model(tf, steps, feature_count, hidden, dropout, learning_rate):
    inputs = tf.keras.Input((steps, feature_count), name="first_35_percent_domain_state")
    encoded = tf.keras.layers.LSTM(hidden, dropout=dropout)(inputs)
    encoded = tf.keras.layers.Dense(max(8, hidden // 2), activation="tanh")(encoded)
    output = tf.keras.layers.Dense(1, activation="tanh", name="recorded_gnt_scaled")(encoded)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
                  loss=tf.keras.losses.Huber(delta=.05), metrics=["mae"])
    return model


def metrics(predicted, actual):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"sample_count": int(error.size), "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2)))}


def plot(output_dir, times, actual, predicted, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use = times >= 500
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(times[use], actual[use], "o-", linewidth=1.8,
              markersize=5, label="GNT")
    axis.plot(times[use], predicted[use], "^--", linewidth=1.8,
              markersize=5, label="Predicted GNT")
    axis.set(xlabel="Time", ylabel="Average GNT")
    axis.set_xticks(times[use]); axis.tick_params(axis="x", rotation=45)
    axis.grid(True, linestyle="--", alpha=.4); axis.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(output_dir / "recorded_gnt_prediction.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "recorded_gnt_prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def run(tf, training_sets, validation_set, args, output_dir):
    """Train on generated inputs and validate against the current averaged input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = validation_set.base_times.astype(np.float64)
    target, sources = recorded_mean_targets(args.raw_gnt_dir, args.runs, times)
    train_features = np.concatenate([observed_domain_features(d) for d in training_sets])
    validation_features = observed_domain_features(validation_set)
    train_target = np.tile(target, len(training_sets)) / recorded.LIMIT
    validation_target = target / recorded.LIMIT
    feature_mean = train_features.mean(axis=(0, 1))
    feature_std = train_features.std(axis=(0, 1))
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    x_train = ((train_features - feature_mean) / feature_std).astype(np.float32)
    x_validation = ((validation_features - feature_mean) / feature_std).astype(np.float32)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(tf, x_train.shape[1], x_train.shape[2],
                        args.window_in_sample_hidden_size, args.dropout, args.learning_rate)
    history = model.fit(
        x_train, train_target[:, None], validation_data=(x_validation, validation_target[:, None]),
        epochs=args.window_in_sample_epochs, batch_size=min(args.batch_size, len(x_train)),
        shuffle=True, verbose=args.verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                patience=args.window_in_sample_patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5,
                patience=max(3, args.window_in_sample_patience // 3), min_lr=1e-5),
        ],
    )
    predicted = np.clip(model.predict(x_validation, verbose=0).ravel(), -1, 1) * recorded.LIMIT
    use = times >= args.evaluation_start
    score = metrics(predicted[use], target[use])
    np.savez_compressed(output_dir / "recorded_gnt_prediction.npz", times=times,
                        actual_recorded_gnt=target, predicted_recorded_gnt=predicted,
                        evaluation_mask=use, feature_mean=feature_mean, feature_std=feature_std)
    with (output_dir / args.history_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump({k: [float(v) for v in values] for k, values in history.history.items()},
                       f, sort_keys=False)
    report = {
        "method": "direct recorded-GNT LSTM from first 7 of 20 window segments",
        "target": "mean recorded GNT from resource_usage.yaml across source runs",
        "training_sets": [d.run for d in training_sets],
        "validation_set": validation_set.run,
        "observed_steps": int(validation_set.observed.shape[1]),
        "training_sample_count": int(len(x_train)),
        "validation_sample_count": int(len(x_validation)),
        "training_and_validation_inputs_are_the_same": False,
        "same_recorded_target_curve_reused_for_generated_inputs": True,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "epochs_ran": int(len(history.history["loss"])),
        "metrics": score,
        "sources": sources,
    }
    with (output_dir / args.summary_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    if args.save_models:
        model.save(output_dir / "recorded_gnt_model.keras")
    plot(output_dir, times, target, predicted, args.figure_dpi)
    print(f"Recorded GNT验证：MAE={score['MAE']:.3f}，RMSE={score['RMSE']:.3f}")
    print(f"输出目录：{output_dir}")
    return report
