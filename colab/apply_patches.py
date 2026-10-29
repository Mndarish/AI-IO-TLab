"""Apply the small Colab-friendly patches to a clone of souvik0306/Air-IO (adaptive_loss_function).

    python colab/apply_patches.py /content/Air-IO

Idempotent: running it twice is harmless. Fails loudly if upstream code changed and a pattern is gone.
"""
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def patch(rel, old, new):
    p = repo / rel
    s = p.read_text()
    if new in s:
        return f"{rel}: already patched"
    assert old in s, f"{rel}: pattern not found (upstream changed?)"
    p.write_text(s.replace(old, new, 1))
    return f"{rel}: patched"


# 1. resuming repeated the last finished epoch (loop starts at `epoch`, ckpt stores the finished one)
print(patch("train_motion.py",
            '            epoch = checkpoint["epoch"]\n',
            '            epoch = checkpoint["epoch"] + 1\n'))
print(patch("train_motion_finetune.py",
            '    epoch = checkpoint["epoch"]\n    best_loss',
            '    epoch = checkpoint["epoch"] + 1\n    best_loss'))

# 2. make sure newest.ckpt is flushed to (Drive-backed) disk before the next epoch starts
print(patch("utils/utils.py",
            '}, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))',
            '}, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))\n    os.sync()'))

# 3. plot legend says "AirIO"; make it configurable (AIIO_LABEL env var, default "AI-IO")
p = repo / "utils/visualize_state.py"
s = p.read_text()
if "AIIO_LABEL" in s:
    print("utils/visualize_state.py: already patched")
else:
    s = s.replace('label="AirIO",', 'label=os.environ.get("AIIO_LABEL", "AI-IO"),', 1)
    s = s.replace('"AirIO_" +', 'os.environ.get("AIIO_LABEL", "AI-IO") + "_" +')
    if not re.search(r"^import os", s, re.M):
        s = "import os\n" + s
    p.write_text(s)
    print("utils/visualize_state.py: patched")
