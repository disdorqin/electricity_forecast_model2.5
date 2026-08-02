import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def FFT_for_Period(x, k=3):
    """
    x: [B, T, C]
    return:
        period_list: [k_eff] (Tensor[int])
        period_weight: [B, k_eff]
    """
    bsz, time_len, _ = x.shape
    # torch.fft 不支持 bfloat16，先转 float32，计算后再转回原 dtype
    orig_dtype = x.dtype
    x_fft = x.float() if orig_dtype == torch.bfloat16 else x
    xf = torch.fft.rfft(x_fft, dim=1)  # [B, F, C]
    freq_amp = torch.abs(xf).mean(0).mean(-1)  # [F]
    if orig_dtype == torch.bfloat16:
        freq_amp = freq_amp.bfloat16()

    if freq_amp.numel() <= 1:
        period = torch.tensor([max(1, time_len)], device=x.device, dtype=torch.long)
        period_weight = torch.ones((bsz, 1), device=x.device, dtype=x.dtype)
        return period, period_weight

    freq_amp = freq_amp.clone()
    freq_amp[0] = 0.0

    k_eff = min(k, freq_amp.numel() - 1)
    if k_eff <= 0:
        k_eff = 1

    _, top_idx = torch.topk(freq_amp, k_eff)
    top_idx = torch.clamp(top_idx, min=1)

    period_list = torch.div(time_len, top_idx, rounding_mode="floor")
    period_list = torch.clamp(period_list, min=1)

    period_weight = torch.abs(xf).mean(-1)[:, top_idx]  # [B, k_eff]
    return period_list, period_weight


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList()
        for i in range(num_kernels):
            kernel_size = 2 * i + 1
            padding = i
            self.kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

    def forward(self, x):
        outputs = []
        for conv in self.kernels:
            outputs.append(conv(x))
        return torch.stack(outputs, dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, top_k=3, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_model, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_model, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        bsz, time_len, channels = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(period_list.shape[0]):
            period = int(period_list[i].item())
            period = max(1, period)

            if time_len % period != 0:
                length = ((time_len // period) + 1) * period
                padding = torch.zeros(
                    (bsz, length - time_len, channels),
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = time_len
                out = x

            out = out.reshape(bsz, length // period, period, channels)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(bsz, -1, channels)
            out = out[:, :time_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)  # [B, T, C, k_eff]
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period_weight, dim=-1)
        return res + x


class SpikeResidualBranch(nn.Module):
    def __init__(self, known_target_len, pred_len, d_model, dropout=0.1):
        super().__init__()
        self.known_target_len = known_target_len
        self.pred_len = pred_len

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.context_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, pred_len)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, target_hist):
        # target_hist: [B, L]
        diff = torch.zeros_like(target_hist)
        if target_hist.size(1) > 1:
            diff[:, 1:] = target_hist[:, 1:] - target_hist[:, :-1]

        mean = target_hist.mean(dim=1, keepdim=True)
        std = target_hist.std(dim=1, keepdim=True) + 1e-6
        z = (target_hist - mean) / std

        diff_mean = diff.mean(dim=1, keepdim=True)
        diff_std = diff.std(dim=1, keepdim=True) + 1e-6
        diff_z = (diff - diff_mean) / diff_std

        soft_spike_mask = torch.sigmoid(torch.abs(z) + torch.abs(diff_z) - 1.0)

        feats = torch.stack([target_hist, diff, z, soft_spike_mask], dim=1)  # [B, 4, L]
        h = self.conv_net(feats)
        spike_context = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        spike_context = self.context_norm(spike_context)
        spike_delta = self.delta_head(spike_context)
        return spike_delta, spike_context


class DynamicPeriodGate(nn.Module):
    def __init__(self, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2 + 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
            nn.Sigmoid(),
        )

    def _period_features(self, target_hist):
        # target_hist: [B, L]
        eps = 1e-8
        centered = target_hist - target_hist.mean(dim=1, keepdim=True)

        xf = torch.fft.rfft(centered, dim=1)
        amp = torch.abs(xf)

        if amp.size(1) > 1:
            amp_no_dc = amp[:, 1:]
            dom = amp_no_dc.max(dim=1).values
            total = amp_no_dc.sum(dim=1) + eps
            dominant_ratio = dom / total

            prob = amp_no_dc / total.unsqueeze(1)
            entropy = -(prob * torch.log(prob + eps)).sum(dim=1)
            entropy = entropy / math.log(amp_no_dc.size(1) + eps)
        else:
            dominant_ratio = torch.zeros(target_hist.size(0), device=target_hist.device, dtype=target_hist.dtype)
            entropy = torch.zeros_like(dominant_ratio)

        if target_hist.size(1) > 1:
            diff = torch.abs(target_hist[:, 1:] - target_hist[:, :-1])
            spike_strength = diff.max(dim=1).values / (diff.mean(dim=1) + eps)
        else:
            spike_strength = torch.zeros(target_hist.size(0), device=target_hist.device, dtype=target_hist.dtype)

        return torch.stack([dominant_ratio, entropy, spike_strength], dim=1)

    def forward(self, base_context, spike_context, target_hist):
        period_features = self._period_features(target_hist)
        gate_input = torch.cat([base_context, spike_context, period_features], dim=-1)
        return self.gate_net(gate_input)


class SpikeGatedTimesNet(nn.Module):
    def __init__(
        self,
        num_variates,
        seq_len,
        pred_len,
        d_model=128,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        dropout=0.1,
        target_index=-1,
        known_target_len=None,
        delta_scale=0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_index = target_index
        self.known_target_len = known_target_len if known_target_len is not None else (seq_len - 2 * pred_len)
        self.delta_scale = float(delta_scale)

        self.embedding = DataEmbedding(num_variates, d_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TimesBlock(seq_len, pred_len, d_model, top_k=top_k, num_kernels=num_kernels) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.spike_branch = SpikeResidualBranch(self.known_target_len, pred_len, d_model, dropout=dropout)
        self.dynamic_gate = DynamicPeriodGate(d_model, pred_len, dropout=dropout)

    def forward(self, x):
        """
        x: [B, seq_len, num_variates]
        """
        enc = self.embedding(x)
        for block in self.blocks:
            enc = block(enc)
            enc = self.norm(enc)

        future_tokens = enc[:, -self.pred_len :, :]
        base_logits = self.out_proj(future_tokens).squeeze(-1)
        base_pred = torch.sigmoid(base_logits)
        base_context = future_tokens.mean(dim=1)

        target_hist = x[:, : self.known_target_len, self.target_index]
        spike_delta, spike_context = self.spike_branch(target_hist)
        spike_delta = self.delta_scale * torch.tanh(spike_delta)
        gate = self.dynamic_gate(base_context, spike_context, target_hist)

        pred = base_pred + gate * spike_delta
        pred = torch.clamp(pred, -0.05, 1.05)
        return pred

