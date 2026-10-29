# Place at: Air-IO/tools/summarize_horizons.py
# Aggregates results/horizons_aiio/*.csv: mean +- std over segments per horizon,
# one row per split (tag). --by_group adds the dynamics-group breakdown
# (Hover / Low / Medium / High / Super-High / Yaw / July).
import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/horizons_aiio")
    ap.add_argument("--mode", default="net", choices=["net", "ekf"])
    ap.add_argument("--which", default="eval")
    ap.add_argument("--include_gaps", action="store_true")
    ap.add_argument("--by_group", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, f"*_{args.which}_{args.mode}.csv")))
    assert files, f"no *_{args.which}_{args.mode}.csv in {args.dir}"
    df = pd.concat([pd.read_csv(p) for p in files])
    df = df[df["mode"] == args.mode]
    if not args.include_gaps:
        df = df[~df["has_gap"]]
    cols = ["rmse_vx", "rmse_vy", "drift_xy", "drift_rate"]
    keys = ["horizon", "tag"] + (["group"] if args.by_group else [])
    g = df.groupby(keys)[cols].agg(["mean", "std"])
    g[("n", "segments")] = df.groupby(keys).size()
    pd.set_option("display.width", 220)
    print(f"mode={args.mode} which={args.which} gaps={'included' if args.include_gaps else 'excluded'}")
    print(g.round(3).to_string())
    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, f"*_{args.which}_{args.mode}_meta.json")))]
    if metas:
        print(pd.DataFrame(metas)[["tag", "network", "params", "latency_ms_bs1", "device"]].to_string(index=False))
    suffix = "_group" if args.by_group else ""
    g.round(4).to_csv(os.path.join(args.dir, f"summary_{args.which}_{args.mode}{suffix}.csv"))


if __name__ == "__main__":
    main()
