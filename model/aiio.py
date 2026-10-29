# Place at: Air-IO/model/aiio.py
#
# AI-IO (arXiv 2603.00597) backbone, without rotor speed, as an Air-IO network.
# Same interface as CodeNetMotionwithRot: forward(data, rot) -> {"net_vel","cov"},
# get_label(), cov = exp(x - 5).
#
#   per-modality Conv1d(k=5) on acc and gyro -> positional encoding ->
#   2x TransformerEncoder (post-norm, relu, dropout 0.2) -> LAST token ->
#   linear heads for body-frame velocity and covariance.
#
# One prediction per window, paired with the label at the window end.
# Attitude is not an input (the `rot` argument is ignored).
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for [B, T, D] tensors."""

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.dtype)


def _conf_get(conf, key, default):
    try:
        return conf.get(key, default)
    except Exception:
        return default


class AIIOTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        sub = int(_conf_get(conf, "aiio_sub_dim", 24))
        nhead = int(_conf_get(conf, "aiio_nhead", 8))
        layers = int(_conf_get(conf, "aiio_layers", 2))
        ff = int(_conf_get(conf, "aiio_ff", 256))
        drop = float(_conf_get(conf, "aiio_dropout", 0.2))
        d_model = 2 * sub

        self.conv_acc = nn.Conv1d(3, sub, 5)
        self.conv_gyro = nn.Conv1d(3, sub, 5)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=drop, batch_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.veldecoder = nn.Linear(d_model, 3)
        self.velcov_decoder = nn.Linear(d_model, 3)
        for m in (self.veldecoder, self.velcov_decoder):
            nn.init.normal_(m.weight, 0, 0.01)

    def cov_decoder(self, x):
        return torch.exp(self.velcov_decoder(x) - 5.0)

    def get_label(self, gt_label):
        return gt_label[:, -1:, :]

    def forward(self, data, rot=None):
        a = self.conv_acc(data["acc"].transpose(-1, -2))
        g = self.conv_gyro(data["gyro"].transpose(-1, -2))
        x = torch.cat([a, g], dim=1).transpose(-1, -2)  # [B, T-4, D]
        x = self.tf(self.pos(x))
        last = x[:, -1, :]
        vel = self.veldecoder(last)[:, None, :]
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(last)[:, None, :]
        return {"cov": cov, "net_vel": vel}
