# AirIO fine-tuning on Vanda: 3 data-scaling experiments (adaptive loss)

| Experiment | Train data |
|---|---|
| `exp0_july_only` | July flights only (no September/October) |
| `exp1_sepoct_20pct` | July + 20% of the Sept+Oct train-eligible flights (11 of 43) |
| `exp2_sepoct_40pct` | July + 40% of the Sept+Oct train-eligible flights (19 of 43) |

Test, eval and inference lists are identical across the three. The 20% set is nested inside the 40% set and is
stratified by dynamics group (every group keeps at least one flight). Hyperparameters, loss
(`normalized_squared_velocity` + CCC + gain + cov), GMM sampling and EuRoC checkpoint are identical.
The exact flights are in `experiments_manifest.csv`.

## 0. Login
```bash
ssh <user>@vanda.nus.edu.sg
```

## 1. Code (one-time)
The adaptive loss lives on Souvik's branch `adaptive_loss_function`, so make sure `~/Air-IO` is on that code.
```bash
# first time only
git clone https://github.com/<your-account>/Air-IO.git ~/Air-IO      # or use your existing clone
cd ~/Air-IO
git remote add souvik https://github.com/souvik0306/Air-IO.git
git fetch souvik
git checkout -b adaptive_loss_function souvik/adaptive_loss_function   # skip if you already have this branch
git log --oneline | head -3        # top commit should be "Gain Loss to fix the issue of under-prediction in low speed"
grep -n "^    loss:" configs/TLab/finetune_motion_body.conf             # should say normalized_squared_velocity
```
Then copy the files from this zip into the repo root (from your laptop):
```bash
scp oct_experiments.zip <user>@vanda.nus.edu.sg:~/Air-IO/
# on Vanda:
cd ~/Air-IO && unzip -o oct_experiments.zip && chmod +x setup_october_data.sh submit_experiments.sh
```

## 2. Data and checkpoint (one-time)
```bash
cd ~/Air-IO
bash setup_airio.sh            # July + September data, EuRoC checkpoint (skips what exists)
bash setup_october_data.sh     # 14th/15th October; prints 23 and 21 flights
ls AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt
```
Run these on the login node, not inside a job. Downloads are about 70 MB each.

## 3. Check the splits before spending GPU time
```bash
cut -d, -f1,2 experiments_manifest.csv | sort | uniq -c
```
Expected train counts: 14, 25, 33. Test is 31 and eval is 26 in all three.

To change fractions or the October split, edit the constants at the top of `make_experiment_configs.py` and run
```bash
module load apptainer/1.4.1
apptainer exec -e /app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif \
    python3 make_experiment_configs.py --data-root .
```

## 4. Submit
```bash
bash submit_experiments.sh                                  # all three as separate GPU jobs
# or one at a time:
qsub -v AIRIO_EXP=exp1_sepoct_20pct airio_finetune_exp.pbs
```
Each job: GMM weights, fine-tune, save orientations, inference, evaluation (24 h walltime, 1 GPU, `auto_free` queue).
Outputs are separated per experiment, so the jobs can run in parallel.

## 5. Monitor
```bash
qstat                       # Q = queued, R = running
qstat -x                    # includes finished jobs
qstat -fx JOB_ID | grep -E 'Resource_List|resources_used|Exit_status'
tail -f airio_exp0.oJOB_ID  # live log (<job name>.o<job id> in ~/Air-IO)
qdel JOB_ID                 # cancel
qstat -Q                    # queue state if jobs sit in Q
```
W&B: project `AirIOtlab_finetune`.

## 6. Where results land
| What | Path |
|---|---|
| Best checkpoint | `experiments/tlab_finetune_oct/finetune_motion_body_<exp>/ckpt/best_model.ckpt` |
| Metrics, plots | `result/<exp>/` |
| Per-flight velocity CSVs | `results_csv/<exp>/` |
| Reference orientations | `orientations/<exp>/` |
```bash
# on your laptop
scp -r <user>@vanda.nus.edu.sg:~/Air-IO/result ./result
scp -r <user>@vanda.nus.edu.sg:~/Air-IO/results_csv ./results_csv
```

## 7. Gotchas
- **Re-running an experiment resumes** from `ckpt/newest.ckpt` if it exists (the job passes `--load_ckpt`; without that flag the script restarts from the EuRoC weights). For a fresh run, delete the experiment folder first.
- `AIRIO_EXP` is required; the job stops immediately with a message if it is missing.
- The GMM cache is per experiment (`gmm_weights/tlab_train_weights_<exp>.npz`) and rebuilds itself if the train list changes.
- Training set size differs between experiments, so steps per epoch differ. Early stopping (patience 6 on the test split) decides the stopping point, so compare on the held-out eval flights, not on epoch counts.
- Report results per dynamics group, especially Hover and Low. The 20%/40% subsets contain only 1 Hover flight, and October has no Hover or Yaw flights.
- Nothing has been excluded for crashes or post-landing segments in the October flights; only the existing 28 July RAW flight_6 window exclusion is applied.
