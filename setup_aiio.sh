#!/bin/bash
# One-time AI-IO setup. Run on the LOGIN node (needs internet; compute jobs should not download).
#   cd ~/Air-IO && unzip -o aiio_vanda.zip && chmod +x *.sh && ./setup_aiio.sh
set -euo pipefail
cd "${AIRIO_REPO_DIR:-$HOME/Air-IO}"
module load apptainer/1.4.1
source ./aiio_env.sh

echo "== 1. code"
echo "branch: $(git branch --show-current 2>/dev/null || echo '?')   commit: $(git log --oneline -1 2>/dev/null || echo '?')"
grep -q "normalized_squared_velocity" model/losses.py || { echo "ERROR: not on the adaptive_loss_function code (no normalized_squared_velocity loss)"; exit 1; }
for f in configs/TLab/finetune_motion_body_exp0_july_only.conf configs/datasets/TLab/tlab_body_exp0_july_only.conf; do
    [ -f "$f" ] || { echo "ERROR: missing $f. Unzip oct_experiments.zip into the repo first."; exit 1; }
done

echo "== 2. lab data"
need=0
for d in T-Lab_28th_July_dataset T-Lab_31st_July_dataset T-Lab_10th_September_dataset; do [ -d "$d" ] || need=1; done
[ $need = 1 ] && bash setup_airio.sh
need=0
for d in T-Lab_14th_October_dataset T-Lab_15th_October_dataset; do [ -d "$d" ] || need=1; done
[ $need = 1 ] && bash setup_october_data.sh
for d in T-Lab_28th_July_dataset T-Lab_31st_July_dataset T-Lab_10th_September_dataset T-Lab_14th_October_dataset T-Lab_15th_October_dataset; do
    echo "$d: $(find "$d" -name imu.csv | wc -l) flights"
done

echo "== 3. EuRoC data (for AI-IO pre-training)"
EUROC_ZIP=https://github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip
if ! find data/euroc -maxdepth 3 -name MH_01_easy 2>/dev/null | grep -q .; then
    mkdir -p data/euroc
    wget -c "$EUROC_ZIP" -O data/euroc/EuRoC-Dataset.zip
    unzip -q -o data/euroc/EuRoC-Dataset.zip -d data/euroc && rm -f data/euroc/EuRoC-Dataset.zip
fi
EUROC_ROOT=$(dirname "$(find data/euroc -maxdepth 3 -name MH_01_easy | head -1)")
echo "EuRoC root: $EUROC_ROOT"
for s in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
    [ -d "$EUROC_ROOT/$s" ] || echo "WARNING: $EUROC_ROOT/$s missing"
done

echo "== 4. configs"
run python3 tools/make_aiio_confs.py --euroc_root "$EUROC_ROOT" ${AIIO_EPOCHS_PRE:+--epochs_pre $AIIO_EPOCHS_PRE} ${AIIO_EPOCHS_FT:+--epochs_ft $AIIO_EPOCHS_FT}

echo "== 5. self-test"
run python3 tools/selftest_aiio.py

echo "== 6. Weights & Biases"
check_wandb

echo "Setup done. Next: bash submit_aiio.sh"
