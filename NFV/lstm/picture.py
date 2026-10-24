"""Plot true raw-log GNT against existing LSTM predictions without retraining.

Preserve the original picture.py ground truth: clip domains before averaging within resource_usage, then average duplicate times and finally across runs. Recomputed window values are available only in explicit diagnostic mode.
Existing LSTM predictions still derive GNT from window traffic; record this target mismatch explicitly in the metrics file."""
from pathlib import Path
import argparse
import numpy as np
import yaml
import lstm_predict as lp

BASE_DIR = Path(__file__).resolve().parent / "GNT"
RAW_BASE_DIR = Path(__file__).resolve().parent.parent / "result_storage" / "EO"
RUNS = [str(run) for run in range(1, 9)]
INPUT_NAME = "lstm_traffic_500_2000.npz"
PREDICTION_NPZ_NAME = "lstm_prediction_500_2000.npz"
SUMMARY_FIGURE_NAME = "lstm_recorded_gnt_comparison.pdf"
PLOT_START_TIME = 500.0
MAX_TIME = 2000.0
FIGURE_DPI = 300
SHOW_SUMMARY_FIGURE = True
Y_MIN = None
Y_MAX = None


def calc_average_temperature(domain_temperatures):
    return lp.average_domain_temperature(domain_temperatures or [])


def load_raw_actual_result(file_path):
    """Restore the original ground-truth curve; never fall back to recomputed window temperatures when reading fails."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else []
    grouped = {}
    for record in records:
        if not isinstance(record, dict) or record.get("time") is None:
            continue
        time = float(record["time"])
        if not np.isfinite(time) or not np.isclose(time % 100., 0.):
            continue
        domains = record.get("domain_temperatures")
        if domains is None:
            domains = record.get("domains")
        if domains is None:
            domains = record.get("temperature_domains")
        grouped.setdefault(time, []).append(calc_average_temperature(domains))
    if not grouped:
        raise ValueError(f"原始日志中没有有效的温度记录：{file_path}")
    times = np.array(sorted(grouped), dtype=float)
    return times, np.array([np.mean(grouped[t]) for t in times], dtype=float)


def load_recorded_actual(raw_base_dir, runs, times):
    actual = []
    for run in runs:
        path = Path(raw_base_dir) / run / "resource_usage.yaml"
        recorded_times, values = load_raw_actual_result(path)
        mapping = dict(zip(recorded_times, values))
        missing = [float(t) for t in times if t not in mapping]
        if missing:
            raise ValueError(f"{path} 缺少真实温度时间点 {missing}，不会用重算值替代。")
        actual.append([mapping[t] for t in times])
    return np.array(actual, dtype=float)


def calculate_prediction_metrics(all_actual, all_predicted):
    """Require complete run/time pairing and report both per-run and mean-curve metrics."""
    actual = np.asarray(all_actual, dtype=float)
    predicted = np.asarray(all_predicted, dtype=float)
    if actual.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError("真实与预测必须为相同形状的实验×时间数组。")
    paired = lp.error_metrics(predicted, actual)
    curve = lp.error_metrics(predicted.mean(axis=0), actual.mean(axis=0))
    return {"per_run_time": paired, "mean_curve": curve}


def load_window_comparison(base_dir, prediction_dir, runs, start, end):
    datasets = lp.load_all_runs(base_dir, runs, INPUT_NAME)
    reference = datasets[runs[0]].base_times
    mask = (reference >= start) & (reference <= end)
    mask[0] = False
    if not np.any(mask):
        raise ValueError("绘图区间内没有非预热窗口。")
    actual, predicted, baseline = [], [], []
    for run in runs:
        data = datasets[run]
        forecast = lp.load_saved_prediction(prediction_dir / run / PREDICTION_NPZ_NAME, data)
        result = lp.evaluate_prediction(data, forecast, start)
        actual.append(lp.gnt_values(result[3])[mask])
        predicted.append(lp.gnt_values(result[4])[mask])
        last = lp.baseline_predictions(data.observed, data.target.shape[1])["last_value"]
        baseline.append(lp.gnt_values(lp.evaluate_prediction(data, last, start)[4])[mask])
    return reference[mask], np.asarray(actual), np.asarray(predicted), np.asarray(baseline)


def draw_all_results(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--model-source", choices=["mean", "recorded", "window"], default="window",
                        help="默认绘制全部8次训练后用前35%输入的窗口拟合。")
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--raw-base-dir", type=Path, default=RAW_BASE_DIR,
                        help="原始真实日志所在的EO目录，其下为1/resource_usage.yaml等。")
    parser.add_argument("--actual-source", choices=["recorded", "window"], default="recorded",
                        help="默认读取resource_usage.yaml中的真实GNT；window仅用于模型内部诊断。")
    parser.add_argument("--show-baseline", action="store_true", help="额外绘制末值保持基线。")
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--start", type=float, default=PLOT_START_TIME)
    parser.add_argument("--end", type=float, default=MAX_TIME)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    if (not args.runs or len(args.runs) != len(set(args.runs))
            or not np.isfinite(args.start) or not np.isfinite(args.end) or args.end < args.start):
        raise ValueError("实验编号不得重复，时间范围须有效。")
    if args.model_source == "mean":
        folder=(args.prediction_dir or args.base_dir/"mean_lstm").resolve()
        with np.load(folder/"predictions.npz") as saved:
            times=saved["times"]; actual=saved["actual_mean"].mean(axis=1)
            predicted=saved["predicted_mean"].mean(axis=1); train=saved["train_mask"]
            validation=saved["validation_mask"]; test=saved["test_mask"]
            mode=str(saved["evaluation_mode"]) if "evaluation_mode" in saved.files else "chronological"
        import matplotlib
        if args.no_show or not SHOW_SUMMARY_FIGURE: matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot(times,actual,"o-",label="Actual mean GNT",linewidth=1.8,markersize=4)
        in_sample=mode == "in_sample"
        ax.plot(times[test],predicted[test],"^--",
                label="LSTM training-set fit" if in_sample else "Selected mean-GNT forecast (future test)",
                linewidth=1.8,markersize=5)
        if not in_sample:
            ax.axvspan(times[train][0]-25,times[train][-1]+25,alpha=.08,color="tab:blue",label="Training period")
            ax.axvspan(times[validation][0]-25,times[validation][-1]+25,alpha=.1,color="tab:orange",label="Validation period")
        ax.set(xlabel="Time",ylabel="Average GNT",
               title="Eight-run mean: training-set fit (not out-of-sample)" if in_sample else
                     "Chronological prediction of the eight-run mean")
        ax.set_xticks(times); ax.tick_params(axis="x",rotation=45); ax.grid(True,linestyle="--",alpha=.4); ax.legend(fontsize=9)
        fig.tight_layout(); output=args.output or folder/"mean_gnt_prediction.png"
        output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=FIGURE_DPI,bbox_inches="tight")
        mae=float(np.mean(np.abs(predicted[test]-actual[test])))
        print(f"8实验平均GNT的{'训练集拟合' if in_sample else '未来测试'}MAE={mae:.3f}；"
              f"时间={times[test].tolist()}")
        print(f"图片：{output}")
        if not args.no_show and SHOW_SUMMARY_FIGURE: plt.show()
        else: plt.close(fig)
        return {"model_source":"mean","test_MAE":mae,"test_times":times[test].tolist()}
    if args.model_source == "recorded":
        folder = (args.prediction_dir or args.base_dir / "recorded_lstm").resolve()
        with np.load(folder / "predictions.npz") as saved:
            actual = saved["actual"]
            predicted = saved["pred_lstm"]
            profile = saved["pred_time_profile"]
            runs, times = saved["runs"].astype(str), saved["times"].astype(float)
        use = np.isin(runs, args.runs) & (times >= args.start) & (times <= args.end)
        if not np.any(use):
            raise ValueError("recorded预测结果中没有指定实验和时间。")
        unique_times = np.unique(times[use])
        def curve(values):
            return np.array([values[use & (times == t)].mean(axis=1).mean() for t in unique_times])
        actual_curve, predicted_curve, profile_curve = curve(actual), curve(predicted), curve(profile)
        report = {"model_source":"recorded","same_target":True,
                  "actual_temperature_target":"recorded_event_gnt",
                  "prediction_temperature_target":"recorded_event_gnt",
                  "lstm":lp.error_metrics(predicted[use].mean(axis=1),actual[use].mean(axis=1)),
                  "mean_curve":lp.error_metrics(predicted_curve,actual_curve)}
        import matplotlib
        if args.no_show or not SHOW_SUMMARY_FIGURE: matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot(unique_times,actual_curve,"o-",label="Actual recorded GNT",linewidth=1.8,markersize=4)
        ax.plot(unique_times,predicted_curve,"^--",label="Validated recorded GNT prediction",linewidth=1.8,markersize=4)
        if args.show_baseline:
            ax.plot(unique_times,profile_curve,"s:",label="Time-profile baseline",linewidth=1.5,markersize=4)
        ax.set(xlabel="Time",ylabel="Average GNT",title="Mean across repeated runs (leave-one-run-out)")
        ax.set_xticks(unique_times); ax.tick_params(axis="x",rotation=45)
        ax.grid(True,linestyle="--",alpha=.4); ax.legend(); fig.tight_layout()
        output=args.output or folder/"recorded_gnt_comparison.png"
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=FIGURE_DPI,bbox_inches="tight")
        with output.with_suffix(".yaml").open("w",encoding="utf-8") as f:
            yaml.safe_dump(report,f,allow_unicode=True,sort_keys=False)
        print(f"同口径recorded GNT：逐次MAE={report['lstm']['MAE']:.3f}；"
              f"平均曲线MAE={report['mean_curve']['MAE']:.3f}")
        print(f"图片：{output}")
        if not args.no_show and SHOW_SUMMARY_FIGURE: plt.show()
        else: plt.close(fig)
        return report
    prediction_dir = args.prediction_dir or args.base_dir / "window_generated_validation"
    validation_result = False
    summary_path = prediction_dir / "lstm_gnt_prediction_summary.yaml"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            saved_summary = yaml.safe_load(f) or {}
        validation_result = (saved_summary.get("evaluation_role") == "validation"
                             or saved_summary.get("validation_set") is not None)
    actual_source = args.actual_source
    direct_path = prediction_dir / "recorded_gnt_prediction.npz"
    averaged_path = prediction_dir / "mean_of_8" / PREDICTION_NPZ_NAME
    direct_recorded = direct_path.exists() and actual_source == "recorded"
    if direct_recorded:
        with np.load(direct_path) as saved:
            all_times = saved["times"].astype(float)
            all_predicted = saved["predicted_recorded_gnt"].astype(float)
        use = (all_times >= args.start) & (all_times <= args.end)
        times = all_times[use]
        actual = load_recorded_actual(args.raw_base_dir, args.runs, times).mean(
            axis=0, keepdims=True,
        )
        predicted = all_predicted[use][None, :]
        previous = np.concatenate(([all_predicted[0]],
                                   load_recorded_actual(args.raw_base_dir, args.runs, all_times).mean(axis=0)[:-1]))
        baseline = previous[use][None, :]
        window_actual = None
        plot_runs = ["mean_of_8"]
    elif averaged_path.exists():
        datasets = lp.load_all_runs(args.base_dir, args.runs, INPUT_NAME)
        averaged = lp.average_run_data(datasets, args.runs)
        forecast = lp.load_saved_prediction(averaged_path, averaged)
        result = lp.evaluate_prediction(averaged, forecast, args.start)
        reference = averaged.base_times
        mask = (reference >= args.start) & (reference <= args.end)
        mask[0] = False
        times = reference[mask]
        window_actual = lp.gnt_values(result[3])[mask][None, :]
        predicted = lp.gnt_values(result[4])[mask][None, :]
        last = lp.baseline_predictions(averaged.observed, averaged.target.shape[1])["last_value"]
        baseline = lp.gnt_values(lp.evaluate_prediction(averaged, last, args.start)[4])[mask][None, :]
        actual = (load_recorded_actual(args.raw_base_dir, args.runs, times).mean(axis=0, keepdims=True)
                  if actual_source == "recorded" else window_actual)
        plot_runs = ["mean_of_8"]
    else:
        times, window_actual, predicted, baseline = load_window_comparison(
            args.base_dir, prediction_dir, args.runs, args.start, args.end,
        )
        actual = (load_recorded_actual(args.raw_base_dir, args.runs, times)
                  if actual_source == "recorded" else window_actual)
        plot_runs = args.runs
    report = {
        "actual_source": actual_source,
        "actual_temperature_target": "recorded_event_gnt" if actual_source == "recorded" else lp.TEMPERATURE_TARGET,
        "prediction_temperature_target": ("recorded_event_gnt" if direct_recorded else lp.TEMPERATURE_TARGET),
        "same_target": direct_recorded or actual_source == "window",
        "actual_files": ([str(args.raw_base_dir / run / "resource_usage.yaml") for run in args.runs]
                         if actual_source == "recorded" else []),
        "temperature_clip": "per_domain_before_mean",
        "prediction_entropy_epsilon": lp.ENTROPY_EPS,
        "undefined_temperature_in_average": 0,
        "runs": plot_runs, "source_runs": args.runs, "times": times.astype(float).tolist(),
        "lstm": calculate_prediction_metrics(actual, predicted),
        "last_value": calculate_prediction_metrics(actual, baseline),
    }
    if window_actual is not None:
        report["window_model_diagnostic"] = calculate_prediction_metrics(window_actual, predicted)
    output = args.output or prediction_dir / (SUMMARY_FIGURE_NAME if actual_source == "recorded"
                                             else "lstm_window_gnt_comparison.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    if args.no_show or not SHOW_SUMMARY_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    actual_curve, predicted_curve = actual.mean(axis=0), predicted.mean(axis=0)
    if not direct_recorded:
        ax.vlines(times,actual_curve,predicted_curve,color="#8c8c8c",alpha=.5,
                  linewidth=1.0,label="Prediction residual")
    ax.plot(times,actual_curve,label="GNT" if actual_source == "recorded" else "Actual window GNT",
            marker="o",linestyle="-",linewidth=1.8,markersize=5,zorder=3)
    if direct_recorded:
        prediction_label = "Predicted GNT"
    elif actual_source == "recorded":
        prediction_label = "Window-model GNT prediction (first 35% input)"
    else:
        prediction_label = ("LSTM validation prediction (first 35% input)"
                            if validation_result else "LSTM training-set fit (first 35% input)")
    ax.plot(times,predicted_curve,label=prediction_label,
            marker="^",linestyle="--",linewidth=1.8,markersize=5,zorder=4)
    if args.show_baseline:
        ax.plot(times,baseline.mean(axis=0),label="Last-value baseline",marker="s",
                linestyle=":",linewidth=1.5,markersize=4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average GNT" if actual_source == "recorded" else "Average window GNT")
    curve_mae=float(np.mean(np.abs(predicted_curve-actual_curve)))
    curve_rmse=float(np.sqrt(np.mean((predicted_curve-actual_curve)**2)))
    if not direct_recorded and actual_source == "recorded":
        ax.text(.02,.03,f"Recorded-target comparison MAE = {curve_mae:.1f}; RMSE = {curve_rmse:.1f}",
                transform=ax.transAxes,fontsize=9,color="#444444")
        ax.set_title("Recorded actual GNT and window-model prediction")
    elif not direct_recorded:
        role = "Validation" if validation_result else "Training-fit"
        ax.text(.02,.03,f"{role} MAE = {curve_mae:.1f}; RMSE = {curve_rmse:.1f}",
                transform=ax.transAxes,fontsize=9,color="#444444")
        ax.set_title(("Generated-set training: current sample validation; connectors show residuals"
                      if validation_result else
                      "Average 8 runs first: training fit; connectors show residuals"))
    ax.set_xticks(times)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)
    if Y_MIN is not None or Y_MAX is not None:
        ax.set_ylim(bottom=Y_MIN, top=Y_MAX)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    with output.with_suffix(".yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    import csv
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        prediction_column = "lstm_recorded_gnt" if direct_recorded else "lstm_window_gnt"
        baseline_column = "last_recorded_gnt" if direct_recorded else "last_value_window_gnt"
        writer.writerow(["run", "time", "actual_recorded_gnt" if actual_source == "recorded" else "actual_window_gnt",
                         prediction_column, baseline_column])
        for i, run in enumerate(plot_runs):
            for j, time in enumerate(times):
                writer.writerow([run, float(time), actual[i,j], predicted[i,j], baseline[i,j]])
    if actual_source == "recorded":
        print(f"真实曲线已恢复：{args.raw_base_dir}/<run>/resource_usage.yaml。")
        if direct_recorded:
            print("预测目标与真实曲线均为resource_usage.yaml中的recorded GNT。")
        else:
            print("现有预测仍来自窗口流量模型；恢复真实曲线并不代表事件GNT预测目标已修正。")
    else:
        print("窗口模型诊断模式：真实值由窗口流量重算，不是原始日志GNT。")
    for name in ("lstm", "last_value"):
        pair, curve = report[name]["per_run_time"], report[name]["mean_curve"]
        print(f"{name}: 逐次MAE={pair['MAE']:.3f} (n={pair['sample_count']}); "
              f"平均曲线MAE={curve['MAE']:.3f}, RMSE={curve['RMSE']:.3f}")
    print(f"图片、指标、逐时刻数据已保存至：{output.parent}")
    if not args.no_show and SHOW_SUMMARY_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return report


if __name__ == "__main__":
    draw_all_results()
