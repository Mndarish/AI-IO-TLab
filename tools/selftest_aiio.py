# Place at: Air-IO/tools/selftest_aiio.py
# 5-second CPU check that the AI-IO network works with Air-IO's loss functions
# and label convention. Run after make_aiio_confs.py (setup_aiio.sh does it).
import glob
import os
import sys

import torch
from pyhocon import ConfigFactory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from model import net_dict
from model.losses import get_motion_loss, get_motion_RMSE

path = sorted(glob.glob("configs/AIIO/finetune_aiio_*.conf"))[0]
conf = ConfigFactory.parse_file(path)
net = net_dict[conf.train.network](conf.train).double()
B, W = 4, 200
data = {"acc": torch.randn(B, W, 3, dtype=torch.double), "gyro": torch.randn(B, W, 3, dtype=torch.double)}
rot = torch.randn(B, W, 3, dtype=torch.double)
out = net(data, rot)
assert out["net_vel"].shape == (B, 1, 3), out["net_vel"].shape
assert out["cov"].shape == (B, 1, 3), out["cov"].shape
label = net.get_label(torch.randn(B, W + 1, 3, dtype=torch.double))
assert label.shape == (B, 1, 3), label.shape
loss = get_motion_loss(out, label, conf.train)
loss["loss"].backward()
assert torch.isfinite(loss["loss"]), loss
rm = get_motion_RMSE(out, label, conf.train)
n = sum(p.numel() for p in net.parameters())
print(f"selftest OK: {conf.train.network}, {n:,} params, loss={loss['loss'].item():.4f}, rmse={rm['loss'].item():.4f}")
