"""Predict the recorded five-domain GNT with a time-profile plus LSTM residual."""
from pathlib import Path
import json
import numpy as np
import gnt_dataset as ds
import gnt_evaluate as ev


def time_profile(datasets, runs):
    times = datasets[runs[0]].times
    if any(not np.array_equal(datasets[r].times, times) for r in runs):
        raise ValueError("All repeated experiments must have the same target times")
    return np.mean([datasets[r].y for r in runs], axis=0)


def residual_training_data(datasets, runs):
    """Each fit run is centered using the other fit runs at the same time."""
    if len(runs) < 2:
        raise ValueError("Residual training requires at least two runs")
    xs, ys = [], []
    for run in runs:
        others = [r for r in runs if r != run]
        xs.append(datasets[run].x)
        ys.append(datasets[run].y - time_profile(datasets, others))
    return np.concatenate(xs), np.concatenate(ys)


def build_model(tf, steps, features, hidden, dropout, learning_rate):
    inputs = tf.keras.Input(shape=(steps, features))
    encoded = tf.keras.layers.LSTM(hidden, dropout=dropout)(inputs)
    encoded = tf.keras.layers.Dense(max(8, hidden//2), activation="tanh")(encoded)
    outputs = tf.keras.layers.Dense(5, kernel_initializer="zeros", bias_initializer="zeros")(encoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
                  loss=tf.keras.losses.Huber(delta=.1))
    return model


def run(args):
    if len(args.runs) < 4 or len(set(args.runs)) != len(args.runs):
        raise ValueError("Recorded-GNT mode requires at least four unique repeated experiments")
    datasets = ds.load_runs(args.raw_gnt_dir, args.runs, args.recorded_history,
                            args.evaluation_start, args.recorded_end)
    output = (args.output_dir or ds.ROOT/"GNT"/"recorded_lstm").resolve()
    if not output.is_relative_to(ds.ROOT):
        raise ValueError("Output must stay inside lstm")
    output.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf
    predictions, actual, run_keys, time_keys, folds = [], [], [], [], []
    profiles, last_values = [], []
    for fold_index, test_run in enumerate(args.runs):
        candidates = [r for r in args.runs if r != test_run]
        rng = np.random.default_rng(args.seed + fold_index)
        validation_run = candidates[int(rng.integers(len(candidates)))]
        fit_runs = [r for r in candidates if r != validation_run]
        train_x, train_residual = residual_training_data(datasets, fit_runs)
        fit_profile = time_profile(datasets, fit_runs)
        val_x = datasets[validation_run].x
        val_residual = datasets[validation_run].y - fit_profile
        mean, std = ds.fit_scaler(train_x)
        train_x, val_x = ds.transform(train_x, mean, std), ds.transform(val_x, mean, std)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold_index)
        model = build_model(tf, train_x.shape[1], train_x.shape[2], args.hidden_size,
                            args.dropout, args.learning_rate)
        initial = model.get_weights()
        zero_val = float(model.evaluate(val_x, val_residual, verbose=0))
        history = model.fit(train_x, train_residual, validation_data=(val_x, val_residual),
                            epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
                            verbose=args.verbose,
                            callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss", patience=args.patience, restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss", patience=max(3,args.patience//3), factor=.5, min_lr=1e-5)])
        trained_val = float(model.evaluate(val_x, val_residual, verbose=0))
        # A residual model is used only when it beats the honest zero-residual profile on validation.
        relative_gain = (zero_val - trained_val) / max(zero_val, 1e-12)
        selected_zero = relative_gain < args.min_residual_improvement
        if selected_zero:
            model.set_weights(initial)
        test_profile = time_profile(datasets, candidates)
        residual = model.predict(ds.transform(datasets[test_run].x, mean, std), verbose=0)
        prediction = np.clip(test_profile + residual, -1., 1.)
        predictions.append(prediction * ds.LIMIT)
        profiles.append(test_profile * ds.LIMIT)
        last_values.append(ds.baselines(datasets[test_run].x)["last_value"] * ds.LIMIT)
        actual.append(datasets[test_run].y * ds.LIMIT)
        run_keys.extend([test_run]*len(datasets[test_run].times))
        time_keys.extend(datasets[test_run].times)
        folds.append({"test_run":test_run,"fit_runs":fit_runs,"validation_run":validation_run,
                      "selected_zero_residual":selected_zero,"zero_validation_loss":zero_val,
                      "trained_validation_loss":trained_val,
                      "relative_validation_improvement":relative_gain,
                      "minimum_required_improvement":args.min_residual_improvement,
                      "epochs_ran":len(history.history["loss"])})
        print(f"Run {test_run}: validation={validation_run}, "
              f"selected={'time profile' if selected_zero else 'time profile + LSTM residual'}")
    actual, predictions = np.concatenate(actual), np.concatenate(predictions)
    profiles, last_values = np.concatenate(profiles), np.concatenate(last_values)
    run_keys, time_keys = np.asarray(run_keys), np.asarray(time_keys)
    forecasts = {"lstm":predictions, "time_profile":profiles, "last_recorded":last_values}
    context = {"mode":"loro","task":"recorded GNT at the same target time",
               "target":"resource_usage.yaml: per-domain T clipped to +/-10000; null=0",
               "method":"leave-one-run-out time profile plus validation-selected LSTM residual",
               "residual_selection_rule":f"use residual only if independent validation loss improves by at least {args.min_residual_improvement:.1%}",
               "scope_warning":"Time profile assumes repeated experiments share the same schedule; it is not future-time extrapolation.",
               "test_labels_used_for_prediction":False,"folds":folds,
               "sources":{r:{"path":d.source,"sha256":d.source_sha256} for r,d in datasets.items()}}
    report = ev.write_results(output, actual, forecasts, run_keys, time_keys, context)
    plot(output, actual, forecasts, run_keys, time_keys, args.figure_dpi)
    print("Recorded GNT metrics:")
    for name, metric in report["metrics"].items():
        print(name, metric["per_run_time_gnt"])
    print(f"Results: {output}")
    return report


def plot(output, actual, forecasts, runs, times, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.unique(times)
    fig, ax = plt.subplots(figsize=(8,5))
    series = [(actual,"Actual recorded GNT","o-"),
              (forecasts["lstm"],"Validated recorded GNT prediction","^--"),
              (forecasts["time_profile"],"Time-profile baseline","s:")]
    for values,label,style in series:
        curve=[values[times==t].mean(axis=1).mean() for t in ts]
        ax.plot(ts,curve,style,label=label,linewidth=1.8,markersize=4)
    ax.set(xlabel="Time",ylabel="Average GNT",title="Mean across repeated runs (leave-one-run-out)")
    ax.set_xticks(ts); ax.tick_params(axis="x",rotation=45); ax.grid(alpha=.3,linestyle="--"); ax.legend()
    fig.tight_layout(); fig.savefig(Path(output)/"recorded_gnt_comparison.png",dpi=dpi,bbox_inches="tight")
    fig.savefig(Path(output)/"recorded_gnt_comparison.pdf",bbox_inches="tight"); plt.close(fig)
    for run in np.unique(runs):
        use = runs == run
        order = np.argsort(times[use])
        run_times = times[use][order]
        fig, ax = plt.subplots(figsize=(8,4.8))
        for values,label,style in series[:2]:
            curve = values[use][order].mean(axis=1)
            ax.plot(run_times,curve,style,label=label,linewidth=1.7,markersize=4)
        ax.set(xlabel="Time",ylabel="Average GNT",title=f"Held-out run {run}")
        ax.set_xticks(run_times); ax.tick_params(axis="x",rotation=45)
        ax.grid(alpha=.3,linestyle="--"); ax.legend(); fig.tight_layout()
        fig.savefig(Path(output)/f"recorded_gnt_run_{run}.png",dpi=dpi,bbox_inches="tight")
        plt.close(fig)
