# AI-IO on T-Lab drone data

[![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/Mndarish/AI-IO-TLab/blob/main/colab/AIIO_colab.ipynb)

Evaluates **AI-IO** (CNN + Transformer inertial odometry, 70k parameters, [arXiv 2603.00597](https://arxiv.org/abs/2603.00597)) as an alternative to AirIO (CNN + GRU) on T-Lab quadrotor flights:
EuRoC pre-training, then fine-tuning on three data-scaling splits (`exp0_july_only`, `exp1_sepoct_20pct`, `exp2_sepoct_40pct`) with the adaptive-loss recipe, and 9 / 12 / 25 s net-only horizon evaluation (velocity RMSE x/y, drift).

This repo is an **overlay**, not a fork. It contains only the new files; the base code is cloned at run time from
[`souvik0306/Air-IO`](https://github.com/souvik0306/Air-IO) (branch `adaptive_loss_function`), which is where the adaptive loss, GMM sampling, datasets and EKF come from. Four tiny upstream patches are applied by `colab/apply_patches.py`.

## Layout
| Path | What |
|---|---|
| `model/aiio.py`, `model/__init__.py` | AI-IO network (`aiio_tf`), registered in `net_dict` (overwrites upstream `model/__init__.py`) |
| `tools/` | `aiio_inference.py`, `eval_horizons.py`, `summarize_horizons.py`, `compare_horizons.py`, `make_aiio_confs.py`, `selftest_aiio.py`, `run_inference.sh` |
| `configs/TLab`, `configs/datasets/TLab`, `experiments_manifest.csv`, `make_experiment_configs.py` | the three splits (same train/test/eval flight lists for all) |
| `setup_aiio.sh`, `aiio_*.pbs`, `submit_aiio.sh` | NUS Vanda (PBS + apptainer) pipeline |
| `colab/AIIO_colab.ipynb` | Colab runner with resumable checkpoints on Google Drive |
| `docs/` | Vanda guides |

## Run on Colab
Click the badge. The notebook clones the base repo, copies this overlay on top, applies the patches, downloads the data and runs the phases. Checkpoints and results are kept on Drive, so it resumes after a disconnect.

## Not included
Raw T-Lab data, EuRoC, checkpoints, W&B logs and results CSVs. Data is downloaded by the setup scripts.

## Credits
AI-IO: SJTU-ViSYS-team ([paper](https://arxiv.org/abs/2603.00597), [code](https://github.com/SJTU-ViSYS-team/AI-IO)); `model/aiio.py` here is a re-implementation without rotor speed. Air-IO: [Air-IO/Air-IO](https://github.com/Air-IO/Air-IO). Adaptive-loss branch and T-Lab data: Souvik. Check the licenses of these projects before reusing.
