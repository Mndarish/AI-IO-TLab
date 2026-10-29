#!/bin/bash
# Inference + metrics + net-only horizons for ONE split, from an existing fine-tuned checkpoint.
# Needs the python deps installed (Colab, or inside the container). Run from the repo root.
#
#   bash tools/run_inference.sh exp1_sepoct_20pct [/path/to/best_model.ckpt]
#
# If a ckpt path is given it is copied to experiments/aiio_oct/finetune_aiio_<exp>/ckpt/best_model.ckpt.
set -euo pipefail
EXP="${1:?usage: run_inference.sh <exp> [ckpt]}"
UPLOADED="${2:-}"
CONFIG="configs/AIIO/finetune_aiio_${EXP}.conf"
DATACONF="configs/datasets/AIIO/tlab_body_aiio_${EXP}.conf"
EXPERIMENT="experiments/aiio_oct/finetune_aiio_${EXP}"
SEQLEN="${AIIO_SEQLEN:-200}"
export AIIO_LABEL="${AIIO_LABEL:-AI-IO}"

if [ -n "$UPLOADED" ]; then
    [ -f "$UPLOADED" ] || { echo "missing $UPLOADED"; exit 1; }
    mkdir -p "$EXPERIMENT/ckpt"
    cp "$UPLOADED" "$EXPERIMENT/ckpt/best_model.ckpt"
fi
[ -f "$EXPERIMENT/ckpt/best_model.ckpt" ] || { echo "no ckpt at $EXPERIMENT/ckpt/best_model.ckpt"; exit 1; }

WB=""; [ "${AIIO_WANDB:-online}" = "online" ] && WB="--wandb_project ${AIIO_WANDB_HORIZON_PROJECT:-AIIO_horizons}"

python3 evaluation/save_ori.py --dataconf "$DATACONF" --savedir "./orientations/aiio_${EXP}" --device cpu
python3 -u tools/aiio_inference.py --config "$CONFIG" --device cuda:0
python3 evaluation/evaluate_motion.py --dataconf "$DATACONF" --exp "$EXPERIMENT" \
    --savedir "./result/aiio_${EXP}" --csvdir "results_csv/aiio_${EXP}" --seqlen "$SEQLEN"
python3 -u tools/eval_horizons.py --config "$CONFIG" --tag "$EXP" --which eval \
    --skip_ekf --stride 12.5 --device cuda:0 $WB
echo "inference finished: $EXP"
