# Place at: Air-IO/tools/eval_horizons.py
#
# Horizon evaluation of AI-IO (9 / 12 / 25 s by default) on the flights of one split.
#
# Flights come from the config's `eval` section (the clean held-out list; `test` is
# used for early stopping) or `test`. For every flight:
#   1. run AI-IO in windows (200 samples, step 4) -> one body-frame velocity / 20 ms,
#   2. cut segments every --stride s (starting >= --warmup s into the flight),
#   3. EKF (EKF/IMUofflinerunner.py, Air-IO's settings) from the GT state at the
#      segment start, run for the longest horizon; 9 s and 12 s are prefixes,
#   4. per horizon: world-frame velocity RMSE in x and y, and drift = horizontal
#      position error at the horizon (and drift / horizon).
# "net" rows: net velocity rotated with GT attitude and dead-reckoned (no EKF).
# Segments that contain an IMU dropout (dt > 20 ms) are flagged has_gap.
#
# Frames: dataset velocity is BODY-frame, so GT world velocity = gt_orientation @
# velocity and the EKF starts from that (the stock runner starts at rest).
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from pyhocon import ConfigFactory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "EKF"))

import datasets  # noqa: F401  (registers Sequence subclasses)
from datasets.dataset import Sequence
from model import net_dict
from utils.utils import make_dataset_save_prefix

# Air-IO's EKF settings (EKF/IMUofflinerunner.py __main__), identical for every run
BIAS_W, INPUT_W, OBS_W = 1e-12, 1e2, 1e-1
GYRO_COV, ACC_COV, R_NO_OBS = 0.006, 0.08, 0.001
GAP_DT = 0.02
GROUPS = ["Super-High-Dynamic", "High-Dynamic", "Medium-Dynamic", "Low-Dynamic", "Hover", "Yaw"]

_G = {}


def group_of(drive):
    parts = drive.split("/")
    for g in GROUPS:
        if g in parts:
            return g
    return "July"


@torch.no_grad()
def run_net(net, seq, W, S, device, bs):
    d = seq.data
    acc, gyro = d["acc"], d["gyro"]
    rotlog = d["gt_orientation"].Log().tensor()
    n = acc.shape[0]
    starts = list(range(0, n - 1 - W, S))
    vel, cov, idx = [], [], []
    for b in range(0, len(starts), bs):
        js = starts[b:b + bs]
        a = torch.stack([acc[j:j + W] for j in js]).to(device)
        g = torch.stack([gyro[j:j + W] for j in js]).to(device)
        r = torch.stack([rotlog[j:j + W] for j in js]).to(device)
        out = net({"acc": a, "gyro": g}, r)
        vel.append(out["net_vel"][:, -1].cpu())
        cov.append(out["cov"][:, -1].cpu() if out["cov"] is not None
                   else torch.full((len(js), 3), 0.01, dtype=torch.double))
        idx += [j + W for j in js]
    return torch.cat(vel), torch.cat(cov), np.array(idx)


def latency_ms(net, seq, W, device, n=100):
    d = seq.data
    rotlog = d["gt_orientation"].Log().tensor()
    ts = []
    with torch.no_grad():
        for k in range(n + 10):
            j = k * 7
            inp = ({"acc": d["acc"][j:j + W][None].to(device), "gyro": d["gyro"][j:j + W][None].to(device)},
                   rotlog[j:j + W][None].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            net(*inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if k >= 10:
                ts.append((time.perf_counter() - t) * 1e3)
    return float(np.mean(ts))


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def run_segment(job):
    from IMUofflinerunner import EKF_runner, SingleIMU
    flight, s, m_max, horizons = job
    F = _G["flights"][flight]
    t, gap_cs = F["t"], F["gap_cs"]
    d = F["seq"].data
    acc, gyro, dt = d["acc"], d["gyro"], d["dt"]
    gt_pos, gt_vw, rot = F["gt_pos"], F["gt_vw"], d["gt_orientation"]
    rows = []

    def row(h, mode, vx, vy, drift, gap):
        return dict(flight=flight, group=F["group"], t0=float(t[s] - t[0]), horizon=h, mode=mode,
                    has_gap=bool(gap), rmse_vx=vx, rmse_vy=vy,
                    rmse_vxy=float(np.sqrt((vx ** 2 + vy ** 2) / 2)),
                    drift_xy=drift, drift_rate=drift / h)

    if not _G["skip_ekf"]:
        state = torch.zeros(15, dtype=torch.float64)
        state[:3] = rot[s:s + 1].Log().tensor()[0]
        state[3:6] = torch.as_tensor(gt_vw[s], dtype=torch.float64)
        state[6:9] = gt_pos[s]
        ekf = EKF_runner(SingleIMU().double())
        ekf.state = state
        obs = F["obs"]
        for i in range(s, m_max):
            q = torch.ones(12, dtype=torch.float64) * BIAS_W
            q[:3] = GYRO_COV
            q[3:6] = ACC_COV * INPUT_W
            r = torch.ones(3, dtype=torch.float64) * R_NO_OBS
            o = None
            if (i + 1) in obs:
                o, c = obs[i + 1]
                r = c * OBS_W
            ekf.run({"gyro": gyro[i], "acc": acc[i], "dt": dt[i]}, observation=o,
                    Q=torch.eye(12, dtype=torch.float64) * q, R=torch.eye(3, dtype=torch.float64) * r)
        est = torch.stack(ekf.est_history).numpy()  # row k -> time index s+k+1
        for h in horizons:
            m_h = min(int(np.searchsorted(t, t[s] + h)), m_max)
            n_r = m_h - s
            ev = est[:n_r, 3:6] - gt_vw[s + 1:m_h + 1]
            dr = np.linalg.norm(est[n_r - 1, 6:8] - gt_pos[m_h, :2].numpy())
            rows.append(row(h, "ekf", _rmse(ev[:, 0]), _rmse(ev[:, 1]), float(dr), gap_cs[m_h] - gap_cs[s] > 0))

    net_idx = F["net_idx"]
    for h in horizons:
        m_h = min(int(np.searchsorted(t, t[s] + h)), m_max)
        sel = (net_idx > s) & (net_idx <= m_h)
        if sel.sum() < 2:
            continue
        ii = net_idx[sel]
        vw = F["net_vw"][sel]
        ev = vw - gt_vw[ii]
        tt = np.concatenate([[t[s]], t[ii]])
        p = gt_pos[s, :2].numpy() + (vw[:, :2] * np.diff(tt)[:, None]).sum(0)
        p += vw[-1, :2] * (t[m_h] - t[ii[-1]])
        dr = np.linalg.norm(p - gt_pos[m_h, :2].numpy())
        rows.append(row(h, "net", _rmse(ev[:, 0]), _rmse(ev[:, 1]), float(dr), gap_cs[m_h] - gap_cs[s] > 0))
    return rows


def log_wandb(args, df, meta):
    try:
        import wandb
        if os.environ.get("WANDB_MODE") == "disabled":
            return
        run = wandb.init(project=args.wandb_project, name=f"{args.tag}_{args.which}_{'ekf' if not args.skip_ekf else 'net'}",
                         group=args.tag, config={**vars(args), **meta}, reinit=True)
        clean = df[~df["has_gap"]]
        for (h, mode), g in clean.groupby(["horizon", "mode"]):
            for m in ("rmse_vx", "rmse_vy", "drift_xy", "drift_rate"):
                run.summary[f"{mode}/h{int(h)}/{m}"] = float(g[m].mean())
            run.summary[f"{mode}/h{int(h)}/segments"] = int(len(g))
        run.log({"segments": wandb.Table(dataframe=df)})
        run.finish()
    except Exception as e:  # never fail the job because of logging
        print("wandb logging skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/AIIO/finetune_aiio_<exp>.conf")
    ap.add_argument("--ckpt", default=None, help="default: <exp_dir>/ckpt/best_model.ckpt")
    ap.add_argument("--tag", required=True, help="e.g. exp1_sepoct_20pct")
    ap.add_argument("--which", default="eval", choices=["eval", "test"])
    ap.add_argument("--max_flights", type=int, default=None)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--horizons", default="9,12,25")
    ap.add_argument("--stride", type=float, default=25.0, help="seconds between segment starts")
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip_ekf", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/horizons_aiio")
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    horizons = [float(h) for h in args.horizons.split(",")]
    hmax = max(horizons)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    conf = ConfigFactory.parse_file(args.config)
    conf_name = os.path.basename(args.config).split(".")[0]
    ckpt = args.ckpt or os.path.join(conf.general.exp_dir, conf_name, "ckpt/best_model.ckpt")
    net = net_dict[conf.train.network](conf.train).to(device).double()
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model_state_dict"])
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())

    dsec = conf.dataset[args.which]
    flights = [(dc.name, dc.data_root, d) for dc in dsec.data_list for d in dc.data_drive][: args.max_flights]
    print(f"[{args.tag}] {conf.train.network} {n_params:,} params, {len(flights)} '{args.which}' flights, ckpt={ckpt}")

    _G["flights"], _G["skip_ekf"] = {}, args.skip_ekf
    jobs, lat = [], None
    for name, root, drive in flights:
        seq = Sequence.subclasses[name](root, drive, coordinate=dsec.get("coordinate", "body_coord"),
                                         mode="infevaluate", gravity=dsec.get("gravity", 9.81007))
        d = seq.data
        fid = make_dataset_save_prefix(root, drive)
        t = d["time"].numpy()
        n = len(t)
        if lat is None:
            lat = latency_ms(net, seq, args.window, device)
        vb, cv, idx = run_net(net, seq, args.window, args.step, device, args.batch)
        rot = d["gt_orientation"]
        gt_vw = (rot @ d["velocity"]).numpy()              # velocity is body-frame
        net_vw = (rot[torch.as_tensor(idx)] @ vb).numpy()  # GT attitude rotates the net output
        gap_cs = np.concatenate([[0], np.cumsum(d["dt"].squeeze(-1).numpy() > GAP_DT)])
        _G["flights"][fid] = dict(seq=seq, t=t, gap_cs=gap_cs, gt_pos=d["gt_translation"], gt_vw=gt_vw,
                                  net_idx=idx, net_vw=net_vw, group=group_of(drive),
                                  obs={int(m): (vb[k], cv[k]) for k, m in enumerate(idx)})
        s = int(np.searchsorted(t, t[0] + args.warmup))
        k0 = len(jobs)
        while True:
            m_max = int(np.searchsorted(t, t[s] + hmax))
            if m_max > n - 2:
                break
            jobs.append((fid, s, m_max, horizons))
            s = int(np.searchsorted(t, t[s] + args.stride))
        print(f"  {fid}: {n} samples, {len(idx)} net outputs, {len(jobs) - k0} segments")

    t0 = time.time()
    if args.workers > 1 and jobs:
        torch.set_num_threads(1)
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(run_segment, jobs, chunksize=1)
    else:
        res = [run_segment(j) for j in jobs]
    df = pd.DataFrame([r for rr in res for r in rr])
    df.insert(0, "tag", args.tag)
    mode = "net" if args.skip_ekf else "ekf"
    csv = os.path.join(args.out, f"{args.tag}_{args.which}_{mode}.csv")
    df.to_csv(csv, index=False)
    meta = {"tag": args.tag, "network": conf.train.network, "params": n_params, "latency_ms_bs1": lat,
            "device": str(device), "n_segments": len(jobs), "eval_seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(args.out, f"{args.tag}_{args.which}_{mode}_meta.json"), "w"), indent=1)
    print(f"[{args.tag}] {len(jobs)} segments, {len(df)} rows, {time.time() - t0:.0f}s -> {csv}")
    if args.wandb_project:
        log_wandb(args, df, meta)


if __name__ == "__main__":
    main()
