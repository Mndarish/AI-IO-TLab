# Place at: Air-IO/tools/aiio_inference.py
#
# Windowed (online) inference for AI-IO over every flight in the config's
# `inference` list. Writes <exp_dir>/net_output.pickle in the same format as
# inference_motion.py, so evaluation/evaluate_motion.py runs on it unchanged.
#
# AI-IO sees 1 s windows (200 samples), advanced by --step samples, and gives one
# body-frame velocity + covariance per window (50 Hz at step 4). This is online
# inference; inference_motion.py runs AirIO over the whole flight at once.
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from pyhocon import ConfigFactory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

import datasets  # noqa: F401  (registers Sequence subclasses)
from datasets.dataset import Sequence
from model import net_dict
from utils import build_dataset_save_prefix


@torch.no_grad()
def run_net(net, seq, W, S, device, bs):
    d = seq.data
    acc, gyro = d["acc"], d["gyro"]
    rotlog = d["gt_orientation"].Log().tensor()  # ignored by AI-IO, kept for the interface
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
    return torch.cat(vel), torch.cat(cov), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="default: <exp_dir>/ckpt/best_model.ckpt")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    conf_name = os.path.basename(args.config).split(".")[0]
    exp_dir = os.path.join(conf.general.exp_dir, conf_name)
    ckpt = args.ckpt or os.path.join(exp_dir, "ckpt/best_model.ckpt")
    device = torch.device(args.device)

    net = net_dict[conf.train.network](conf.train).to(device).double()
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    print(f"loaded {ckpt} (epoch {ck['epoch']}), {sum(p.numel() for p in net.parameters()):,} params")

    dconf = conf.dataset.inference
    result, t_net, n_win = {}, 0.0, 0
    for data_conf in dconf.data_list:
        for path in data_conf.data_drive:
            seq = Sequence.subclasses[data_conf.name](
                data_conf.data_root, path, coordinate=dconf.get("coordinate", "body_coord"),
                mode="infevaluate", gravity=dconf.get("gravity", 9.81007))
            t0 = time.time()
            vel, cov, idx = run_net(net, seq, args.window, args.step, device, args.batch)
            t_net += time.time() - t0
            n_win += len(idx)
            ts = seq.data["time"][idx][:, None]            # [N,1], exact dataset timestamps
            result[build_dataset_save_prefix(data_conf, path)] = {
                "net_vel": vel, "cov": cov[None], "ts": ts}
            print(f"  {path}: {len(idx)} outputs")

    out = os.path.join(exp_dir, "net_output.pickle")
    with open(out, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    json.dump({"windows": n_win, "seconds": t_net, "ms_per_window_batched": 1e3 * t_net / max(n_win, 1),
               "device": str(device)}, open(os.path.join(exp_dir, "aiio_inference_meta.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
