# Sourced by the AI-IO PBS jobs and setup_aiio.sh (do not run directly).
IMAGE="${AIRIO_IMAGE:-/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif}"
run()     { apptainer exec -e "$IMAGE" "$@"; }
run_gpu() { apptainer exec --nv -e "$IMAGE" "$@"; }

# ---- Weights & Biases -------------------------------------------------------
# AIIO_WANDB = online (default) | offline | off
#   online : needs `wandb login` done once (credentials in ~/.netrc)
#   offline: logs to ./wandb, upload later with `wandb sync wandb/offline-run-*`
#   off    : no W&B at all (adds --log, which this repo uses to disable it)
# apptainer -e clears the environment, so W&B settings are passed with APPTAINERENV_.
WANDB_REQ="${AIIO_WANDB:-online}"
LOGFLAG=""
case "$WANDB_REQ" in
    off)     LOGFLAG="--log" ;;
    offline) export APPTAINERENV_WANDB_MODE=offline ;;
    online)  export APPTAINERENV_WANDB_MODE=online ;;
    *) echo "AIIO_WANDB must be online, offline or off (got '$WANDB_REQ')"; exit 1 ;;
esac

check_wandb() {
    [ "$WANDB_REQ" = "online" ] || { echo "W&B mode: $WANDB_REQ"; return 0; }
    [ "${AIIO_WANDB_CHECK:-1}" = "1" ] || return 0
    if run python3 - <<'PY'
import netrc, sys
try:
    ok = "api.wandb.ai" in netrc.netrc().hosts
except Exception:
    ok = False
print("W&B credentials in ~/.netrc:", ok)
sys.exit(0 if ok else 1)
PY
    then return 0; fi
    echo "ERROR: no W&B login found. On the login node run:"
    echo "  module load apptainer/1.4.1"
    echo "  apptainer exec -e $IMAGE python3 -m wandb login"
    echo "or submit with AIIO_WANDB=offline (or off)."
    exit 1
}
