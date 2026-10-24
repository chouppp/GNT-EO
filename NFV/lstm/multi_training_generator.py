"""Generate multiple averaged synthetic training sets; preserve current NPZs for validation."""
from pathlib import Path
import numpy as np
import traffic_generator as tg


def generate(datasets, runs, count, seed, output_dir,
             trend_ratio=.02, wave_ratio=.01, link_noise_std=.01,
             ar_coefficient=.95, max_perturbation=.03):
    if count < 2:
        raise ValueError("At least two generated training sets are required")
    reference=datasets[runs[0]]
    baselines={r:datasets[r].complete.mean(axis=1,dtype=np.float64) for r in runs}
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for stale_path in output_dir.glob("training_set_*.npz"):
        stale_path.unlink()
    generated=[]
    for set_index in range(count):
        run_flows=[]
        set_seed=seed+set_index*1000
        for run_index,run in enumerate(runs):
            run_flows.append(tg.augment_baseline_flows(
                baseline=baselines[run],steps_per_window=reference.complete.shape[1],
                seed=set_seed+run_index,trend_ratio=trend_ratio,wave_ratio=wave_ratio,
                link_noise_std=link_noise_std,ar_coefficient=ar_coefficient,
                max_perturbation=max_perturbation))
        flow=np.mean(run_flows,axis=0).astype(np.float32)
        observed_steps=reference.observed.shape[1]
        path=output_dir/f"training_set_{set_index+1:03d}.npz"
        np.savez_compressed(path,link_flow=flow,
                            observed_link_flow=flow[:,:observed_steps,:],
                            target_link_flow=flow[:,observed_steps:,:],
                            base_times=reference.base_times,
                            edge_sources=reference.edge_sources,edge_targets=reference.edge_targets,
                            generation_seed=np.asarray(set_seed),source_runs=np.asarray(runs))
        # Reuse data types already loaded by the caller to avoid importing the training entrypoint.
        generated.append(type(reference)(
            run=f"generated_{set_index+1:03d}",path=path,
            observed=flow[:,:observed_steps,:],target=flow[:,observed_steps:,:],
            complete=flow,base_times=reference.base_times.copy(),
            edge_sources=reference.edge_sources.copy(),edge_targets=reference.edge_targets.copy()))
    return generated
