# AI-IO on Vanda: 3 data-scaling splits, same recipe as your AirIO runs

Trains the AI-IO network (CNN + Transformer, acc + gyro only) on EuRoC, then fine-tunes it on the same three splits as
`oct_experiments` (`exp0_july_only`, `exp1_sepoct_20pct`, `exp2_sepoct_40pct`) with the same adaptive loss, GMM sampling,
learning rates and flight lists, and evaluates on the held-out `eval` flights. AirIO is not part of this.


## 0. One-time install (login node)
```bash
# laptop
scp aiio_vanda.zip <user>@vanda.nus.edu.sg:~/Air-IO/
# Vanda
cd ~/Air-IO
git branch --show-current          # adaptive_loss_function (same code as your AirIO runs)
unzip -o aiio_vanda.zip && chmod +x *.sh
```
`oct_experiments.zip` must already be unpacked in the repo (the AI-IO configs are generated from its configs).

## 1. Weights & Biases (once)
```bash
module load apptainer/1.4.1
apptainer exec -e /app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif python3 -m wandb login
```
Every job checks for the login first and stops with a message if it is missing. Modes (`AIIO_WANDB=...` at submit time):
`online` (default), `offline` (logs to `./wandb`, upload later with `wandb sync wandb/offline-run-*`), `off`.

| What | W&B project | Run name |
|---|---|---|
| EuRoC pre-training | `AirIOaiio_euroc` | `aiio_tf` |
| Fine-tuning | `AIIO_tlab_oct` | `finetune_aiio_<exp>_<timestamp>` |
| 9/12/25 s horizon results | `AIIO_horizons` | `<exp>_eval_net`, `<exp>_eval_ekf` |

## 2. Setup (login node)
```bash
cd ~/Air-IO
./setup_aiio.sh
```
Downloads EuRoC (and any lab day still missing), writes the AI-IO configs, runs a CPU self-test (network + Air-IO loss
shapes) and checks the W&B login. Optional: `AIIO_EPOCHS_PRE=100 AIIO_EPOCHS_FT=30 ./setup_aiio.sh` to set epochs.
It should print flights `train 14 / 25 / 33`, `test 31`, `eval 26` for the three splits.

## 3. Submit
```bash
bash submit_aiio.sh
```
Chains: `aiio_pretrain` (EuRoC) -> `aiio_exp` x3 (fine-tune, inference, Air-IO metrics, net-only horizons) -> `aiio_ekf` x3
(EKF horizons, 34 workers). Options: `SKIP_PRETRAIN=1`, `AIIO_EKF=0`, `AIIO_WANDB=offline`.
Single job: `qsub -v AIIO_EXP=exp1_sepoct_20pct aiio_exp.pbs`.
Quick smoke test: `AIIO_EPOCHS_PRE=2 AIIO_EPOCHS_FT=2 ./setup_aiio.sh`, then `AIIO_EKF=0 bash submit_aiio.sh`
(re-run setup with the real epochs afterwards and delete `experiments/aiio_*`).

## 4. Monitor
```bash
qstat
tail -f "$(ls -t aiio_*.o* | head -1)"
```

## 5. Results
| What | Path |
|---|---|
| EuRoC checkpoint | `experiments/aiio_euroc/aiio_tf/ckpt/best_model.ckpt` |
| Fine-tuned checkpoints | `experiments/aiio_oct/finetune_aiio_<exp>/ckpt/best_model.ckpt` |
| Air-IO metrics (ATE, velocity RMSE, RTE per flight) | `result/aiio_<exp>/result.json`, `results_csv/aiio_<exp>/` |
| Horizon rows (per segment) | `results/horizons_aiio/<exp>_eval_{net,ekf}.csv` |
```bash
IMG=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif
apptainer exec -e $IMG python3 tools/summarize_horizons.py --mode net --by_group
apptainer exec -e $IMG python3 tools/summarize_horizons.py --mode ekf
```
Horizon metrics (9, 12, 25 s): world-frame velocity RMSE in x and y, and drift = horizontal position error at the horizon
(plus drift / horizon). Gap-free segments by default; add `--include_gaps` to include IMU-dropout segments.
Compare against your AirIO runs by joining `result/<exp>/result.json` and `result/aiio_<exp>/result.json` on the flight name,
restricted to the `eval` flights in `experiments_manifest.csv`.

## 6. Things that differ from the AirIO runs (read before comparing)
- **Loss.** `normalized_squared_velocity` + covariance term, same weights. `ccc_weight` and `gain_weight` are 0, because they
  are computed over the time axis of each window and AI-IO outputs a single velocity per window. Use `--seq_terms` in
  `tools/make_aiio_confs.py` to keep them (CCC then contributes nothing; the gain term becomes a per-sample ratio).
- **Online vs offline.** AI-IO sees 1 s windows (step 4, 50 Hz outputs). AirIO's `inference_motion.py` runs the whole flight at
  once. The same `evaluate_motion.py` scores both, but AI-IO is the harder protocol.
- **Pre-training.** AI-IO has no public EuRoC checkpoint (its released one needs rotor speed), so it is trained from scratch
  on EuRoC with 200-sample windows and AirIO's EuRoC recipe (lr 1e-3, batch 128, 100 epochs).
- **`test` vs `eval`.** `test` drives early stopping and checkpoint choice, so the horizon tables use the `eval` list only.
- **Attitude.** `evaluate_motion.py` rotates velocity with integrated attitude (like your AirIO runs). The horizon script uses GT
  attitude for the net-only rows and the EKF starts from GT state at each segment start.
- **Gaps.** Training keeps your branch's behaviour (no gap mask). Horizon segments touching a dropout (dt > 20 ms) are flagged.
- Hover and Low are the groups to look at (`--by_group`); the 20% / 40% subsets contain a single Hover flight.

## 7. Overrides
`AIRIO_IMAGE`, `AIIO_WANDB`, `AIIO_SEQLEN`, `AIIO_EKF_WORKERS`, `AIIO_EKF_STRIDE` (s), `AIIO_EKF_MAX_FLIGHTS`,
`AIIO_WANDB_CHECK=0` (skip the login check), `AIIO_WANDB_HORIZON_PROJECT`.
