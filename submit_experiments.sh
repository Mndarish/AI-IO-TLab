#!/bin/bash
# Submit the three data-scaling experiments as independent GPU jobs.
set -euo pipefail
cd "${AIRIO_REPO_DIR:-$HOME/Air-IO}"
for exp in exp0_july_only exp1_sepoct_20pct exp2_sepoct_40pct; do
    qsub -N "airio_${exp%%_*}" -v AIRIO_EXP="$exp" airio_finetune_exp.pbs
done
