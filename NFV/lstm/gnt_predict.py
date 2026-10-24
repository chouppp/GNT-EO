"""Predict raw five-domain log GNT directly; default to three history windows and whole-run holdouts."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import gnt_dataset as ds
import gnt_evaluate as ev


def fit_ridge(x, y, alpha=10.):
    x=x.reshape(len(x),-1).astype(np.float64)
    design=np.column_stack([x,np.ones(len(x))])
    penalty=np.eye(design.shape[1])*alpha
    penalty[-1,-1]=0.
    return np.linalg.solve(design.T@design+penalty,design.T@y)


def predict_ridge(x, weights):
    return np.clip(np.column_stack([x.reshape(len(x),-1),np.ones(len(x))])@weights,-1.,1.)


def configure_validation(folds, validation_on_train=False):
    """Explicit diagnostic mode allows matching training/validation runs but always excludes test runs."""
    configured=[]
    for original in folds:
        fold={**original}
        if validation_on_train:
            fold["unused_validation_runs"]=list(original["validation"])
            fold["validation"]=list(original["fit"])
        if set(fold["test"]) & (set(fold["fit"]) | set(fold["validation"])):
            raise ValueError("Test runs must remain separate from training and validation")
        fold["validation_is_independent"]=not validation_on_train
        configured.append(fold)
    return configured


def build_model(tf, history, features, hidden, rate, mean_weight):
    inputs=tf.keras.Input(shape=(history,features))
    encoded=tf.keras.layers.LSTM(hidden)(inputs)
    outputs=tf.keras.layers.Dense(5,activation="tanh",kernel_initializer="zeros",bias_initializer="zeros")(encoded)
    model=tf.keras.Model(inputs,outputs)
    def domain_and_mean_loss(y_true,y_pred):
        error=y_pred-y_true
        return tf.reduce_mean(tf.square(error),axis=-1)+mean_weight*tf.square(tf.reduce_mean(error,axis=-1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rate,clipnorm=1.),loss=domain_and_mean_loss)
    return model


def train_fold(tf, datasets, fold, args, out):
    train_x,train_y=ds.concatenate(datasets,fold["fit"])
    val_x,val_y=ds.concatenate(datasets,fold["validation"])
    # Fit parameters only on training runs; exclude test data from scale estimation.
    mean,std=ds.fit_scaler(train_x)
    x=ds.transform(train_x,mean,std)
    vx=ds.transform(val_x,mean,std)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model=build_model(tf,x.shape[1],x.shape[-1],args.hidden_size,args.learning_rate,args.mean_loss_weight)
    initial_weights=model.get_weights()
    initial_val=float(model.evaluate(vx,val_y,verbose=0))
    history=model.fit(x,train_y,validation_data=(vx,val_y),epochs=args.epochs,
        batch_size=args.batch_size,verbose=args.verbose,shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=args.patience,restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=max(3,args.patience//3),min_lr=1e-5)])
    final_val=float(model.evaluate(vx,val_y,verbose=0))
    # Checkpoint selection never uses test error. Diagnostic val_loss comes from training data, not independent validation.
    selected_initial=initial_val<=final_val
    if selected_initial: model.set_weights(initial_weights)
    ridge=fit_ridge(x,train_y,args.ridge_alpha)
    out.mkdir(parents=True,exist_ok=True)
    model.save(out/"gnt_model.keras")
    np.savez_compressed(out/"preprocessing.npz",mean=mean,std=std,ridge_weights=ridge)
    info={**fold,"train_samples":len(x),"validation_samples":len(vx),"epochs_ran":len(history.history["loss"]),
          "best_epoch":0 if selected_initial else int(np.argmin(history.history["val_loss"])+1),
          "selected_initial_zero_checkpoint":selected_initial,"initial_validation_loss":initial_val,
          "selected_validation_loss":initial_val if selected_initial else final_val,
          "history":{k:[float(v) for v in values] for k,values in history.history.items()}}
    if args.validation_on_train:
        train_runs=np.concatenate([np.repeat(r,len(datasets[r].y)) for r in fold["fit"]])
        train_times=np.concatenate([datasets[r].times for r in fold["fit"]])
        fitted={**ds.baselines(train_x),"lstm":model.predict(x,verbose=0),"ridge":predict_ridge(x,ridge)}
        fitted={n:p*ds.LIMIT for n,p in fitted.items()}
        fit_output=out/"training_fit"
        fit_report=ev.write_results(fit_output,train_y*ds.LIMIT,fitted,train_runs,train_times,
            {"mode":"in_sample","evaluation_scope":"training samples used in fitting and checkpoint selection",
             "validation_is_independent":False,"fit_runs":fold["fit"],"test_runs":fold["test"]})
        ev.plot_results(fit_output,train_y*ds.LIMIT,fitted,train_runs,train_times,"in_sample")
        info["in_sample_fit_metrics"]=fit_report["metrics"]
        info["in_sample_fit_directory"]=str(fit_output)
    (out/"training.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    actual,run_keys,time_keys=[],[],[]
    predictions={name:[] for name in ("lstm","ridge","zero","last_value","history_mean")}
    for run in fold["test"]:
        d=datasets[run]
        tx=ds.transform(d.x,mean,std)
        forecasts={**ds.baselines(d.x),"lstm":model.predict(tx,verbose=0),"ridge":predict_ridge(tx,ridge)}
        for name,pred in forecasts.items(): predictions[name].append(pred*ds.LIMIT)
        actual.append(d.y*ds.LIMIT)
        run_keys.extend([run]*len(d.y))
        time_keys.extend(d.times.tolist())
    return np.concatenate(actual),{n:np.concatenate(p) for n,p in predictions.items()},np.array(run_keys),np.array(time_keys),info


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-base-dir",type=Path,default=ds.RAW_BASE)
    p.add_argument("--task",choices=["history","prefix35"],default="history",
                   help="prefix35 requires genuine timestamped observations; never uses synthetic substeps")
    p.add_argument("--observation-dir",type=Path,default=ds.ROOT/"GNT/observations")
    p.add_argument("--check-data-only",action="store_true",help="Validate data without training or writing results")
    p.add_argument("--runs",nargs="+",default=[str(i) for i in range(1,9)])
    p.add_argument("--mode",choices=["holdout","loro"],default="holdout")
    p.add_argument("--validation-on-train",action="store_true",
                   help="拟合诊断：训练组同时用作验证组，测试组仍独立；另存训练拟合图。")
    p.add_argument("--train-runs",nargs="+")
    p.add_argument("--validation-runs",nargs="+")
    p.add_argument("--test-runs",nargs="+")
    p.add_argument("--history",type=int,default=3)
    p.add_argument("--start",type=float,default=500.)
    p.add_argument("--end",type=float,default=2000.)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=15)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--hidden-size",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=.001)
    p.add_argument("--mean-loss-weight",type=float,default=1.)
    p.add_argument("--ridge-alpha",type=float,default=10.)
    p.add_argument("--seed",type=int,default=20260910)
    p.add_argument("--verbose",type=int,choices=[0,1,2],default=2)
    p.add_argument("--output-dir",type=Path,default=None)
    return p.parse_args()


def main():
    args=parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.hidden_size,args.history)<1:
        raise ValueError("Training sizes must be positive")
    if (not all(np.isfinite(v) for v in (args.learning_rate,args.mean_loss_weight,args.ridge_alpha))
            or args.learning_rate<=0 or args.mean_loss_weight<0 or args.ridge_alpha<=0):
        raise ValueError("Invalid loss/optimizer parameters")
    folds=ds.make_folds(args.runs,args.mode,args.train_runs,args.validation_runs,args.test_runs)
    folds=configure_validation(folds,args.validation_on_train)
    output_name=args.mode+("_train_validation_diagnostic" if args.validation_on_train else "")
    if args.task == "prefix35": output_name="prefix35_"+output_name
    output=(args.output_dir or ds.ROOT/"GNT/direct_gnt"/output_name).resolve()
    if not output.is_relative_to(ds.ROOT): raise ValueError("Output must stay inside lstm")
    # Validate data before loading the training framework or writing results.
    if args.task == "prefix35":
        import gnt_prefix
        datasets=gnt_prefix.load_runs(args.raw_base_dir,args.observation_dir,args.runs,args.start,args.end)
    else:
        datasets=ds.load_runs(args.raw_base_dir,args.runs,args.history,args.start,args.end)
    if args.check_data_only:
        print(f"Validated task={args.task}: {sum(len(d.y) for d in datasets.values())} samples; "
              f"input shape={next(iter(datasets.values())).x.shape[1:]}")
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    import tensorflow as tf
    actual,all_runs,all_times,details=[],[],[],[]
    forecasts={n:[] for n in ("lstm","ridge","zero","last_value","history_mean")}
    if args.validation_on_train:
        print("诊断模式：训练与验证使用相同样本，val_loss不代表独立预测误差；测试组保持独立。",flush=True)
    for fold in folds:
        print(f"Fold {fold['name']}: fit={fold['fit']}, validation={fold['validation']}, test={fold['test']}",flush=True)
        a,p,r,t,info=train_fold(tf,datasets,fold,args,output/"models"/fold["name"])
        actual.append(a);all_runs.append(r);all_times.append(t);details.append(info)
        for n in forecasts:forecasts[n].append(p[n])
    actual=np.concatenate(actual);runs=np.concatenate(all_runs);times=np.concatenate(all_times)
    forecasts={n:np.concatenate(p) for n,p in forecasts.items()}
    context={"mode":args.mode,"task":("first 35% real observations -> recorded window GNT" if args.task == "prefix35"
                                                    else "past recorded windows -> next recorded five-domain GNT"),
             "validation_is_independent":not args.validation_on_train,
             "checkpoint_selection":"training samples (diagnostic)" if args.validation_on_train else "independent validation runs",
             "evaluation_scope":"held-out test runs; excluded from training and checkpoint selection",
             "is_35_percent_observation_task":args.task == "prefix35","target":"mean of per-record clipped domain T; null=0",
             "scaling":"target T/10000; input scaler fit on training runs only",
             "history_length":next(iter(datasets.values())).x.shape[1],"feature_names":ds.FEATURE_NAMES,
             "config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             "folds":details,"evaluated_runs":np.unique(runs).tolist(),
             "mean_curve_warning":"Cross-fold means can cancel errors; primary metrics are paired run/time and domain errors.",
             "sources":{r:{"path":d.source,"sha256":d.source_sha256} for r,d in datasets.items()}}
    report=ev.write_results(output,actual,forecasts,runs,times,context)
    ev.plot_results(output,actual,forecasts,runs,times,args.mode)
    for name,value in report["metrics"].items():
        print(name, value["per_run_time_gnt"],flush=True)
    print(f"Results: {output}",flush=True)


if __name__=="__main__": main()
