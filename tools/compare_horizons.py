"""Side-by-side horizon table (one column per split), overall and by dynamics group.

    python tools/compare_horizons.py [--dir results/horizons_aiio]
"""
import argparse
import os

import pandas as pd

EXPS = ["exp0_july_only", "exp1_sepoct_20pct", "exp2_sepoct_40pct"]
COLS = ["rmse_vx", "rmse_vy", "drift_xy"]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/horizons_aiio")
a = ap.parse_args()

frames = []
for e in EXPS:
    f = os.path.join(a.dir, f"{e}_eval_net.csv")
    if os.path.isfile(f):
        frames.append(pd.read_csv(f))
    else:
        print("missing", f)
df = pd.concat(frames)
df = df[(df["mode"] == "net") & (~df["has_gap"])]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def side(keys):
    return df.groupby(keys + ["tag"])[COLS].mean().round(3).unstack("tag")


print("\n=== OVERALL (mean over eval segments) ===")
print(side(["horizon"]).to_string())
print("\n=== BY GROUP ===")
print(side(["horizon", "group"]).to_string())
print("\nsegments per split:\n", df.groupby(["horizon", "tag"]).size().unstack("tag").to_string())
side(["horizon"]).to_csv(os.path.join(a.dir, "compare_3exp_overall.csv"))
side(["horizon", "group"]).to_csv(os.path.join(a.dir, "compare_3exp_by_group.csv"))
