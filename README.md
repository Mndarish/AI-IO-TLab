# AI-IO on T-Lab drone data

[![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/Mndarish/AI-IO-TLab/blob/main/colab/AIIO_run_all.ipynb)

Evaluates **AI-IO** (CNN + Transformer inertial odometry, 70k parameters, [arXiv 2603.00597](https://arxiv.org/abs/2603.00597)) as an alternative to AirIO (CNN + GRU) on T-Lab quadrotor flights: EuRoC pre-training, then fine-tuning on three data-scaling splits (`exp0_july_only`, `exp1_sepoct_20pct`, `exp2_sepoct_40pct`) with the adaptive-loss recipe, and 9 / 12 / 25 s net-only horizon evaluation (velocity RMSE x/y, drift).

## Quick start (Colab)
1. Click the badge above.
2. Runtime → Run all. Allow Google Drive access when asked, and choose "Run anyway" on the not-authored-by-Google warning.
3. Wait. Checkpoints and results are saved to `MyDrive/AIIO_colab`. If Colab disconnects or the GPU quota ends, reconnect and Run all again; it resumes from the last finished epoch.

Needs a GPU runtime (T4 is enough). Optional: add a Colab secret `WANDB_API_KEY` to log to Weights & Biases online; without it runs log offline to `MyDrive/AIIO_colab/wandb` (upload later with `wandb sync`).

The notebook pre-trains on EuRoC for 25 epochs, fine-tunes from the epoch-24 checkpoint on each split, and prints the horizon tables. Seeds are not fixed, so numbers will be close to, but not identical with, the reported ones.

## How the repo works
This repo is an **overlay**, not a fork. It holds only the new files. The notebook clones the base code at run time from [`souvik0306/Air-IO`](https://github.com/souvik0306/Air-IO) (branch `adaptive_loss_function`: adaptive loss, GMM sampling, datasets, EKF), copies this repo on top, and applies four small patches (`colab/apply_patches.py`): resume off-by-one fix in both training scripts, checkpoint flush to Drive, and a plot-legend label.

| Path | What |
|---|---|
| `model/aiio.py`, `model/__init__.py` | AI-IO network (`aiio_tf`), registered in `net_dict` (overwrites the upstream `model/__init__.py`) |
| `tools/` | `aiio_inference.py`, `eval_horizons.py`, `summarize_horizons.py`, `compare_horizons.py`, `make_aiio_confs.py`, `selftest_aiio.py`, `run_inference.sh` |
| `configs/TLab`, `configs/datasets/TLab`, `experiments_manifest.csv`, `make_experiment_configs.py` | the three splits (identical test and eval flights across splits) |
| `setup_aiio.sh`, `setup_october_data.sh`, `aiio_*.pbs`, `submit_aiio.sh`, `submit_experiments.sh` | setup and the NUS Vanda (PBS + apptainer) pipeline |
| `colab/` | `AIIO_run_all.ipynb`, `apply_patches.py`, shims that let the Vanda scripts run on Colab |
| `docs/` | Vanda guides |

## Not included
Raw T-Lab data, EuRoC, checkpoints, W&B logs and result CSVs. The setup scripts download the data.

## Credits
- AI-IO: SJTU-ViSYS-team ([paper](https://arxiv.org/abs/2603.00597), [code](https://github.com/SJTU-ViSYS-team/AI-IO)). `model/aiio.py` here is a re-implementation without rotor speed.
- Air-IO: [Air-IO/Air-IO](https://github.com/Air-IO/Air-IO).
- Adaptive-loss branch and T-Lab data: Souvik ([souvik0306](https://github.com/souvik0306)).

Check the licenses of these projects before reusing.
