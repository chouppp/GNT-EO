"""Unified evaluation of raw five-domain GNT predictions; prioritize per-run errors, with mean curves as supplementary views."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from gnt_dataset import LIMIT, ROOT


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape or not actual.size or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(actual)):
        raise ValueError("Metrics require finite, matching, nonempty pairs")
    diff = predicted-actual
    return {"count":int(diff.size), "MAE":float(np.abs(diff).mean()), "RMSE":float(np.sqrt(np.mean(diff**2)))}


def evaluate(actual, forecasts, runs, times):
    """Inputs are raw GNT, not the training values divided by 10000."""
    runs, times = np.asarray(runs), np.asarray(times)
    if actual.shape != (len(runs),5) or len(times)!=len(runs):
        raise ValueError("Expected samples x 5 domains with run/time keys")
    if len(set(zip(runs.tolist(),times.tolist())))!=len(runs):
        raise ValueError("Duplicate prediction keys")
    result = {}
    unique_times = np.unique(times)
    for name, p in forecasts.items():
        domain = metrics(p,actual)
        gnt, true = p.mean(axis=1), actual.mean(axis=1)
        counts = [set(runs[times==t]) for t in unique_times]
        if any(s!=counts[0] for s in counts):
            raise ValueError("Mean curves require the same experiments at every time")
        result[name] = {
            "per_run_time_gnt":metrics(gnt,true), "domain_pooled":domain,
            "per_domain":{str(d):metrics(p[:,d],actual[:,d]) for d in range(5)},
            "per_run":{r:metrics(gnt[runs==r],true[runs==r]) for r in np.unique(runs)},
            "mean_curve_auxiliary":metrics(np.array([gnt[times==t].mean() for t in unique_times]),
                                            np.array([true[times==t].mean() for t in unique_times])),
        }
    return result


def write_results(output, actual, forecasts, runs, times, context):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Outputs must stay inside the lstm directory")
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate(actual, forecasts, runs, times)
    np.savez_compressed(output/"predictions.npz", actual=actual, runs=runs, times=times,
                        **{f"pred_{name}":p for name,p in forecasts.items()})
    with (output/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["run","time","domain","actual_recorded_T_clipped",*forecasts])
        for i,(r,t) in enumerate(zip(runs,times)):
            for d in range(5): writer.writerow([r,float(t),d,actual[i,d],*[p[i,d] for p in forecasts.values()]])
    report={**context,"metrics":result}
    (output/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def plot_results(output, actual, forecasts, runs, times, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    # Clearly label training-fit diagnostics; do not present them as held-out predictions.
    for run in [*np.unique(runs), "mean"]:
        use=np.ones(len(runs),bool) if run=="mean" else runs==run
        ts=np.unique(times[use])
        fig,ax=plt.subplots(figsize=(8,4.8))
        series={"Actual recorded GNT":actual, "Direct GNT LSTM":forecasts["lstm"],
                "Ridge baseline":forecasts["ridge"]}
        for (label,p),style in zip(series.items(),["o-","^--","s:"]):
            values=[p[use & (times==t)].mean(axis=1).mean() for t in ts]
            ax.plot(ts,values,style,label=label,linewidth=1.6,markersize=4)
        ax.set(xlabel="Target time",ylabel="Average GNT")
        title=(f"Recorded GNT: run {run}" if run!="mean" else
               f"{'Cross-validation (auxiliary)' if mode=='loro' else 'Held-out'} mean: runs {', '.join(np.unique(runs))}")
        if mode=="in_sample":
            title=(f"Training fit (NOT held out): run {run}" if run!="mean" else
                   f"Training fit (NOT held out): runs {', '.join(np.unique(runs))}")
        ax.set_title(title)
        ax.set_xticks(ts)
        ax.tick_params(axis="x",rotation=45)
        ax.grid(alpha=.3,linestyle="--")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output/f"gnt_{run}.png",dpi=160)
        fig.savefig(output/f"gnt_{run}.pdf")
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir",type=Path,required=True)
    a=p.parse_args()
    folder=a.result_dir.resolve()
    if not folder.is_relative_to(ROOT): raise ValueError("Result directory must be inside lstm")
    with np.load(folder/"predictions.npz") as z:
        actual,runs,times=z["actual"],z["runs"],z["times"]
        forecasts={k[5:]:z[k] for k in z.files if k.startswith("pred_")}
    context=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    context.pop("metrics",None)
    write_results(folder,actual,forecasts,runs,times,context)
    plot_results(folder,actual,forecasts,runs,times,context["mode"])
    print(f"Re-evaluated saved predictions: {folder}")


if __name__=="__main__": main()
