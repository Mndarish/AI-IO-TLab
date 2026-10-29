#!/usr/bin/env python3
"""Generate the 3 data-scaling fine-tuning experiments (configs + manifest).

exp0_july_only    : train = July flights only (no September/October data)
exp1_sepoct_20pct : July + 20% of the September+October train-eligible flights
exp2_sepoct_40pct : July + 40% of the September+October train-eligible flights

test / eval / inference lists are IDENTICAL in all three, so results are comparable.
Subsets are nested (20% is contained in 40%) and stratified by dynamics group.
Run from the repo root:  python make_experiment_configs.py --data-root .
"""
import argparse, csv, glob, math, os, random
from pyhocon import ConfigFactory

BASE_DATA = "configs/datasets/TLab/tlab_body.conf"
BASE_CFG = "configs/TLab/finetune_motion_body.conf"
OCT_DAYS = {"14th": "T-Lab_14th_October_dataset", "15th": "T-Lab_15th_October_dataset"}
GROUPS = ["High-Dynamic", "Low-Dynamic", "Medium-Dynamic"]
OCT_SPLIT = {"train": {1, 3, 5}, "test": {2}, "eval": {4}}   # by flight number
EXPS = [("exp0_july_only", 0.0), ("exp1_sepoct_20pct", 0.2), ("exp2_sepoct_40pct", 0.4)]
SEED = 0


def entry_text(e, drives, extra=""):
    return ("        {name: %s\n        window_size: %s\n        step_size: %s\n"
            "        data_root: %s\n        data_drive: [%s]\n%s        }"
            % (e["name"], e["window_size"], e["step_size"], e["data_root"], ", ".join(drives), extra))


def oct_flights(root, day, grp, src):
    d = os.path.join(root, OCT_DAYS[day], grp, src)
    return sorted(int(os.path.basename(p).split("_")[1]) for p in glob.glob(d + "/flight_*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    a = ap.parse_args()
    base = ConfigFactory.parse_file(BASE_DATA)
    ds = base["dataset"] if "dataset" in base else base

    # ---- inventory of every flight: (split, root, drive) ----
    fixed = {"train": [], "test": [], "eval": [], "inference": []}   # entries kept verbatim (July + Sept)
    for sec in fixed:
        for e in ds[sec].data_list:
            fixed[sec].append(e)

    pool = []      # Sept + Oct train-eligible flights: (root, drive, group, src)
    oct_by_sec = {s: [] for s in ["train", "test", "eval", "inference"]}
    for e in fixed["train"]:
        if "September" in e["data_root"]:
            for d in e["data_drive"]:
                g, s, _ = d.split("/")
                pool.append((e["data_root"], d, g, s))
    for day in OCT_DAYS:
        root = "./" + OCT_DAYS[day]
        for g in GROUPS:
            for s in ["AI", "RAW"]:
                for n in oct_flights(a.data_root, day, g, s):
                    d = f"{g}/{s}/flight_{n}"
                    oct_by_sec["inference"].append((root, d))
                    for sec, ids in OCT_SPLIT.items():
                        if n in ids:
                            oct_by_sec[sec].append((root, d))
                            if sec == "train":
                                pool.append((root, d, g, s))

    # ---- nested, stratified ranking inside each dynamics group ----
    rng = random.Random(SEED)
    ranking = {}
    for g in sorted({p[2] for p in pool}):
        strata = {}
        for p in sorted(pool):
            if p[2] == g:
                strata.setdefault((p[0], p[3]), []).append(p)
        for k in strata:
            rng.shuffle(strata[k])
        order, keys = [], sorted(strata)
        while any(strata.values()):
            for k in keys:
                if strata[k]:
                    order.append(strata[k].pop())
        ranking[g] = order

    def chosen(frac):
        out = []
        for g, order in ranking.items():
            k = math.ceil(frac * len(order)) if frac > 0 else 0
            out += order[:k]
        return set((p[0], p[1]) for p in out)

    os.makedirs("configs/datasets/TLab", exist_ok=True)
    manifest = []
    for name, frac in EXPS:
        sel = chosen(frac)
        sections = []
        for sec in ["train", "test", "eval", "inference"]:
            blocks = []
            for e in fixed[sec]:
                is_sep = "September" in e["data_root"]
                drives = list(e["data_drive"])
                if sec == "train" and is_sep:
                    drives = [d for d in drives if (e["data_root"], d) in sel]
                    if not drives:
                        continue
                extra = ""
                if "exclude_windows" in e:
                    raise SystemExit("exclude_windows present: extend generator")
                blocks.append(entry_text(e, drives))
            for root in sorted({r for r, _ in oct_by_sec[sec]}):
                drives = [d for r, d in oct_by_sec[sec] if r == root]
                if sec == "train":
                    drives = [d for d in drives if (root, d) in sel]
                if drives:
                    blocks.append(entry_text(fixed[sec][0], drives).replace(
                        fixed[sec][0]["data_root"], root))
            sections.append((sec, blocks))
        raw = open(BASE_DATA).read()
        out = raw
        import re
        for sec, blocks in sections:
            m = re.search(rf"^{sec}:\s*\{{", out, re.M)
            lst = out.index("data_list", m.start())
            lb = out.index("[", lst)
            rb = out.index("\n    ]", lb)
            out = out[:lb + 1] + "\n" + ",\n".join(blocks) + out[rb:]
        # per-experiment orientation path so parallel jobs never collide
        out = re.sub(r"rot_path:\s*\S+", f"rot_path: ./orientations/{name}/orientation_output.pickle", out)
        open(f"configs/datasets/TLab/tlab_body_{name}.conf", "w").write(out)

        cfg = open(BASE_CFG).read()
        cfg = cfg.replace("../datasets/TLab/tlab_body.conf", f"../datasets/TLab/tlab_body_{name}.conf")
        cfg = cfg.replace("gmm_weights/tlab_train_weights.npz", f"gmm_weights/tlab_train_weights_{name}.npz")
        cfg = re.sub(r"exp_dir:\s*\S+", "exp_dir: experiments/tlab_finetune_oct", cfg)
        open(f"configs/TLab/finetune_motion_body_{name}.conf", "w").write(cfg)

        c = ConfigFactory.parse_file(f"configs/TLab/finetune_motion_body_{name}.conf")
        for sec in ["train", "test", "eval"]:
            for e in c.dataset[sec].data_list:
                for d in e.data_drive:
                    manifest.append((name, sec, e.data_root, d))
        print(name, {s: sum(len(e.data_drive) for e in c.dataset[s].data_list) for s in ["train", "test", "eval"]})

    with open("experiments_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["experiment", "split", "data_root", "flight"]); w.writerows(manifest)
    print("pool (Sept+Oct train-eligible):", len(pool), {g: len(o) for g, o in ranking.items()})


if __name__ == "__main__":
    main()
