#!/bin/bash
# Submit the whole AI-IO study with job dependencies:
#   pre-train on EuRoC -> fine-tune + evaluate each split -> EKF horizons per split
#
#   bash submit_aiio.sh                      # everything
#   SKIP_PRETRAIN=1 bash submit_aiio.sh      # EuRoC checkpoint already trained
#   AIIO_EKF=0 bash submit_aiio.sh           # skip the EKF jobs
#   AIIO_WANDB=offline bash submit_aiio.sh   # online (default) | offline | off
set -euo pipefail
cd "$(dirname "$0")"

EXPS=($(ls configs/AIIO/finetune_aiio_*.conf | sed 's#.*/finetune_aiio_##; s#\.conf$##'))
[ ${#EXPS[@]} -gt 0 ] || { echo "No configs/AIIO/*.conf: run ./setup_aiio.sh first"; exit 1; }
VARS="AIIO_WANDB=${AIIO_WANDB:-online}"

dep=()
if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
    J0=$(qsub -v "$VARS" aiio_pretrain.pbs)
    echo "pretrain         $J0"
    dep=(-W "depend=afterok:$J0")
fi
for e in "${EXPS[@]}"; do
    J=$(qsub "${dep[@]}" -v "$VARS,AIIO_EXP=$e" aiio_exp.pbs)
    echo "exp $e  $J"
    if [ "${AIIO_EKF:-1}" = "1" ]; then
        K=$(qsub -W "depend=afterok:$J" -v "$VARS,AIIO_EXP=$e" aiio_ekf.pbs)
        echo "ekf $e  $K"
    fi
done
echo "monitor: qstat   |   logs: aiio_*.o<jobid>"
