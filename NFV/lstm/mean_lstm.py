"""Chronological prediction of the GNT curve averaged over repeated runs."""
from pathlib import Path
import csv
import json
import numpy as np
import gnt_dataset as ds
import gnt_evaluate as ev


def average_runs(datasets, runs):
    reference = datasets[runs[0]]
    if any(not np.array_equal(datasets[r].times, reference.times) for r in runs):
        raise ValueError("All runs must share the same time axis")
    return reference.times.copy(), np.mean([datasets[r].x for r in runs], axis=0), \
           np.mean([datasets[r].y for r in runs], axis=0)


def chronological_masks(times, train_end, validation_end):
    train = times <= train_end
    validation = (times > train_end) & (times <= validation_end)
    test = times > validation_end
    if train.sum() < 4 or validation.sum() < 1 or test.sum() < 1:
        raise ValueError("Chronological split needs >=4 train, >=1 validation and >=1 test samples")
    return train, validation, test


def build_model(tf, steps, features, hidden, dropout, learning_rate):
    inputs=tf.keras.Input(shape=(steps,features))
    encoded=tf.keras.layers.LSTM(hidden,dropout=dropout)(inputs)
    outputs=tf.keras.layers.Dense(5,kernel_initializer="zeros",bias_initializer="zeros")(encoded)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate,clipnorm=1.),
                  loss=tf.keras.losses.Huber(delta=.1))
    return model


def run(args):
    datasets=ds.load_runs(args.raw_gnt_dir,args.runs,args.recorded_history,
                          args.mean_start,args.recorded_end)
    times,x,y=average_runs(datasets,args.runs)
    if args.mean_evaluation == "in_sample":
        train=val=test=np.ones(len(times),dtype=bool)
    else:
        train,val,test=chronological_masks(times,args.mean_train_end,args.mean_validation_end)
    baseline=x[:,-1,10:15].copy()
    mean,std=ds.fit_scaler(x[train])
    scaled=ds.transform(x,mean,std)
    import tensorflow as tf
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(args.seed)
    hidden=args.mean_hidden_size if args.mean_evaluation == "in_sample" else args.hidden_size
    epochs=args.mean_epochs if args.mean_evaluation == "in_sample" else args.epochs
    patience=args.mean_patience if args.mean_evaluation == "in_sample" else args.patience
    model=build_model(tf,x.shape[1],x.shape[2],hidden,args.dropout,args.learning_rate)
    initial=model.get_weights()
    train_residual=y[train]-baseline[train]
    val_residual=y[val]-baseline[val]
    initial_loss=float(model.evaluate(scaled[val],val_residual,verbose=0))
    history=model.fit(scaled[train],train_residual,validation_data=(scaled[val],val_residual),
                      epochs=epochs,batch_size=min(args.batch_size,int(train.sum())),
                      shuffle=False,verbose=args.verbose,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=patience,
                                     restore_best_weights=True),
                                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,
                                     patience=max(3,patience//3),min_lr=1e-5)])
    trained_loss=float(model.evaluate(scaled[val],val_residual,verbose=0))
    trained_prediction=np.clip(baseline+model.predict(scaled,verbose=0),-1.,1.)
    train_mean=np.broadcast_to(y[train].mean(axis=0),y.shape).copy()
    candidates={"trained_lstm":trained_prediction,"last_average":baseline,
                "train_mean":train_mean,"zero":np.zeros_like(y)}
    validation_mae={name:float(np.mean(np.abs(values[val]-y[val])))
                    for name,values in candidates.items()}
    best_baseline=min(("last_average","train_mean","zero"),key=validation_mae.get)
    baseline_loss=validation_mae[best_baseline]
    gain=(baseline_loss-validation_mae["trained_lstm"])/max(baseline_loss,1e-12)
    selected_method=("trained_lstm" if args.mean_evaluation == "in_sample" else
                     "trained_lstm" if gain >= args.min_residual_improvement else best_baseline)
    prediction=candidates[selected_method]
    output=(args.output_dir or ds.ROOT/"GNT"/"mean_lstm").resolve()
    if not output.is_relative_to(ds.ROOT): raise ValueError("Output must stay inside lstm")
    output.mkdir(parents=True,exist_ok=True)
    test_forecasts={"selected":prediction[test]*ds.LIMIT,
                    "trained_lstm":trained_prediction[test]*ds.LIMIT,
                    "last_average":baseline[test]*ds.LIMIT,
                    "train_mean":train_mean[test]*ds.LIMIT,
                    "zero":np.zeros_like(y[test])*ds.LIMIT}
    actual_test=y[test]*ds.LIMIT
    keys=np.repeat("mean_of_8",test.sum())
    report=ev.evaluate(actual_test,test_forecasts,keys,times[test])
    context={"mode":("training_set_fit" if args.mean_evaluation == "in_sample" else "chronological_mean_sequence"),
             "task":"past mean states -> future GNT averaged over eight repeated runs",
             "runs":args.runs,"train_times":times[train].tolist(),
             "validation_times":times[val].tolist(),"test_times":times[test].tolist(),
             "test_labels_used_for_training_or_selection":args.mean_evaluation == "in_sample",
             "evaluation_is_out_of_sample":args.mean_evaluation != "in_sample",
             "selected_method":selected_method,
             "validation_domain_mae_normalized":validation_mae,
             "initial_validation_loss":initial_loss,"trained_validation_loss":trained_loss,
             "relative_validation_improvement":gain,
             "minimum_required_improvement":args.min_residual_improvement,
             "epochs_ran":len(history.history["loss"]),
             "sample_size_warning":("Metrics are training-set fit and do not measure generalization."
                                    if args.mean_evaluation == "in_sample" else
                                    "Only one averaged sequence is available; test metrics cover future times only."),
             "metrics":report,
             "sources":{r:{"path":d.source,"sha256":d.source_sha256} for r,d in datasets.items()}}
    np.savez_compressed(output/"predictions.npz",times=times,actual_mean=y*ds.LIMIT,
                        predicted_mean=prediction*ds.LIMIT,trained_lstm=trained_prediction*ds.LIMIT,
                        last_average=baseline*ds.LIMIT,train_mean=train_mean*ds.LIMIT,
                        train_mask=train,validation_mask=val,test_mask=test,
                        evaluation_mode=np.asarray(args.mean_evaluation))
    (output/"metrics.json").write_text(json.dumps(context,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f); writer.writerow(["time","split","domain","actual_mean_gnt","predicted_mean_gnt"])
        for i,t in enumerate(times):
            split="train" if train[i] else "validation" if val[i] else "test"
            for domain in range(5): writer.writerow([t,split,domain,y[i,domain]*ds.LIMIT,prediction[i,domain]*ds.LIMIT])
    model.save(output/"mean_gnt_model.keras")
    np.savez_compressed(output/"preprocessing.npz",mean=mean,std=std)
    plot(output,times,y*ds.LIMIT,prediction*ds.LIMIT,train,val,test,args.figure_dpi)
    metric=report["selected"]["per_run_time_gnt"]
    label="training fit" if args.mean_evaluation == "in_sample" else "chronological test"
    print(f"Mean-GNT {label}: MAE={metric['MAE']:.3f}, RMSE={metric['RMSE']:.3f}")
    print(f"Selected method={selected_method}; results={output}")
    return context


def plot(output,times,actual,predicted,train,val,test,dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actual_curve=actual.mean(axis=1); predicted_curve=predicted.mean(axis=1)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(times,actual_curve,"o-",label="Actual mean GNT",linewidth=1.8,markersize=4)
    in_sample=np.array_equal(train,test) and np.all(train)
    ax.plot(times[test],predicted_curve[test],"^--",
            label="LSTM training-set fit" if in_sample else "Selected mean-GNT forecast (future test)",
            linewidth=1.8,markersize=5)
    if not in_sample:
        ax.axvspan(times[train][0]-25,times[train][-1]+25,alpha=.08,color="tab:blue",label="Training period")
        ax.axvspan(times[val][0]-25,times[val][-1]+25,alpha=.10,color="tab:orange",label="Validation period")
    ax.set(xlabel="Time",ylabel="Average GNT",
           title="Eight-run mean: training-set fit (not out-of-sample)" if in_sample else
                 "Chronological prediction of the eight-run mean")
    ax.set_xticks(times); ax.tick_params(axis="x",rotation=45); ax.grid(alpha=.3,linestyle="--"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(Path(output)/"mean_gnt_prediction.png",dpi=dpi,bbox_inches="tight")
    fig.savefig(Path(output)/"mean_gnt_prediction.pdf",bbox_inches="tight"); plt.close(fig)
