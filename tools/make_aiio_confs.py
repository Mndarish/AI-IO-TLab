# Place at: Air-IO/tools/make_aiio_confs.py
#
# Builds every AI-IO config from files that already exist in the repo:
#   configs/datasets/EuRoC/Euroc_body_aiio.conf   EuRoC data, 200-sample windows
#   configs/EuRoC/aiio_tf.conf                    EuRoC pre-training (AirIO's EuRoC recipe)
#   configs/datasets/AIIO/tlab_body_aiio_<exp>.conf   copy of your split, own orientation dir
#   configs/AIIO/finetune_aiio_<exp>.conf         your fine-tune recipe with network: aiio_tf
# <exp> = every configs/TLab/finetune_motion_body_<exp>.conf found (exp0/exp1/exp2).
# Train / test / eval flight lists are copied verbatim, so the splits are exactly
# the ones used for your AirIO runs.
import argparse
import glob
import os
import re

from pyhocon import ConfigFactory

AIIO_PARAMS = """   aiio_sub_dim: 24
   aiio_nhead: 8
   aiio_layers: 2
   aiio_ff: 256
   aiio_dropout: 0.2
"""


def sub1(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"could not find '{what}' in the source config (pattern {pattern!r})")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc_root", required=True)
    ap.add_argument("--epochs_pre", type=int, default=100)
    ap.add_argument("--epochs_ft", type=int, default=None, help="default: keep the value in your config (30)")
    ap.add_argument("--seq_terms", action="store_true",
                    help="keep ccc_weight / gain_weight (they need several outputs per window; "
                         "AI-IO gives one, so they are switched off by default)")
    ap.add_argument("--wandb_project", default="AIIO_tlab_oct")
    args = ap.parse_args()

    # ---- EuRoC pre-training
    src = open("configs/datasets/EuRoC/Euroc_body.conf").read()
    src = src.replace("DATA_ROOT", os.path.abspath(args.euroc_root))
    src = re.sub(r"window_size:\s*1000", "window_size: 200", src)
    open("configs/datasets/EuRoC/Euroc_body_aiio.conf", "w").write(src)

    pre = f"""general:
{{
   exp_dir: experiments/aiio_euroc
}}

dataset:
{{
   include "../datasets/EuRoC/Euroc_body_aiio.conf"
   collate: {{type: motion}}
}}

train:
{{
   network: aiio_tf
{AIIO_PARAMS}
   lr: 1e-3
   min_lr: 1e-5
   batch_size: 128
   max_epoches: {args.epochs_pre}
   patience: 5
   factor: 0.2
   weight_decay: 1e-4

   weight: 1e2
   cov_weight: 1e-4
   propcov: True
   covaug: True
   loss: Huber_loss005
   rotloss: Huber_loss005

   save_freq: 5
   eval_freq: 1
}}
"""
    os.makedirs("configs/EuRoC", exist_ok=True)
    open("configs/EuRoC/aiio_tf.conf", "w").write(pre)
    pretrained = "./experiments/aiio_euroc/aiio_tf/ckpt/best_model.ckpt"

    # ---- lab fine-tuning, one per split
    os.makedirs("configs/AIIO", exist_ok=True)
    os.makedirs("configs/datasets/AIIO", exist_ok=True)
    found = sorted(glob.glob("configs/TLab/finetune_motion_body_exp*.conf"))
    assert found, "no configs/TLab/finetune_motion_body_exp*.conf found (unzip oct_experiments first)"
    for path in found:
        exp = os.path.basename(path)[len("finetune_motion_body_"):-len(".conf")]
        ds_src = f"configs/datasets/TLab/tlab_body_{exp}.conf"
        ds = open(ds_src).read()
        ds = ds.replace(f"./orientations/{exp}/", f"./orientations/aiio_{exp}/")
        open(f"configs/datasets/AIIO/tlab_body_aiio_{exp}.conf", "w").write(ds)

        t = open(path).read()
        t = t.replace(f"../datasets/TLab/tlab_body_{exp}.conf", f"../datasets/AIIO/tlab_body_aiio_{exp}.conf")
        t = sub1(r"^\s*exp_dir:.*$", "   exp_dir: experiments/aiio_oct", t, "exp_dir")
        t = sub1(r"^\s*wandb_project:.*$", f"   wandb_project: {args.wandb_project}", t, "wandb_project")
        t = sub1(r"^\s*network:\s*codewithrot\s*$", "   network: aiio_tf\n" + AIIO_PARAMS.rstrip("\n"), t, "network")
        t = sub1(r"^\s*pretrained_ckpt:.*$", f"   pretrained_ckpt: {pretrained}", t, "pretrained_ckpt")
        t = re.sub(r"(cache_path:\s*)gmm_weights/\S+", rf"\1gmm_weights/aiio_{exp}.npz", t)
        if not args.seq_terms:
            t = sub1(r"^(\s*ccc_weight:).*$", r"\1 0.0", t, "ccc_weight")
            t = sub1(r"^(\s*gain_weight:).*$", r"\1 0.0", t, "gain_weight")
        if args.epochs_ft is not None:
            t = sub1(r"^(\s*max_epoches:).*$", rf"\1 {args.epochs_ft}", t, "max_epoches")
        out = f"configs/AIIO/finetune_aiio_{exp}.conf"
        open(out, "w").write(t)

        c = ConfigFactory.parse_file(out)
        n = {s: sum(len(e.data_drive) for e in c.dataset[s].data_list) for s in ("train", "test", "eval")}
        print(f"{exp}: flights {n}  network={c.train.network}  loss={c.train.loss} "
              f"ccc={c.train.ccc_weight} gain={c.train.gain_weight} epochs={c.train.max_epoches}")
    print("EuRoC pre-train conf: configs/EuRoC/aiio_tf.conf ->", pretrained)


if __name__ == "__main__":
    main()
