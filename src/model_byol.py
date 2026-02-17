import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFeatureEncoder(nn.Module):
    """時間方向をできるだけ保持するため、時間方向はstride=1中心で畳み込みするエンコーダ。"""

    def __init__(self, in_ch: int = 1, hidden_ch: int = 64, feat_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch * 2, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(hidden_ch * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch * 2, hidden_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv1d(hidden_ch * 2, feat_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)  # [B, C, F', T]
        h = h.mean(dim=2)  # 周波数軸のみ平均化して時間軸Tは維持
        z = self.proj(h)
        return F.normalize(z, dim=1)  # [B, D, T]


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLModel(nn.Module):
    """BYOL/BYOL-A共通で利用する自己教師ありモデル。"""

    def __init__(
        self,
        in_ch: int = 1,
        encoder_hidden: int = 64,
        feat_dim: int = 128,
        projector_hidden: int = 256,
        predictor_hidden: int = 256,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.online_encoder = TemporalFeatureEncoder(in_ch, encoder_hidden, feat_dim)
        self.online_projector = MLPHead(feat_dim, projector_hidden)
        self.online_predictor = MLPHead(feat_dim, predictor_hidden)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for po, pt in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            pt.data = self.ema_decay * pt.data + (1.0 - self.ema_decay) * po.data
        for po, pt in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            pt.data = self.ema_decay * pt.data + (1.0 - self.ema_decay) * po.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)

    def _forward_online(self, x: torch.Tensor) -> torch.Tensor:
        z = self.online_encoder(x)
        p = self.online_predictor(self.online_projector(z))
        return F.normalize(p, dim=1)

    @torch.no_grad()
    def _forward_target(self, x: torch.Tensor) -> torch.Tensor:
        z = self.target_encoder(x)
        z = self.target_projector(z)
        return F.normalize(z, dim=1)

    def byol_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        p1 = self._forward_online(x1)
        p2 = self._forward_online(x2)
        with torch.no_grad():
            t1 = self._forward_target(x1)
            t2 = self._forward_target(x2)

        loss_12 = 2.0 - 2.0 * (p1 * t2).sum(dim=1).mean()
        loss_21 = 2.0 - 2.0 * (p2 * t1).sum(dim=1).mean()
        return 0.5 * (loss_12 + loss_21)


class BYOLAugment:
    """スペクトログラム向け簡易Augmentation。BYOL/BYOL-Aを設定で切替。"""

    def __init__(self, mode: str = "byol", noise_std: float = 0.02, time_mask_ratio: float = 0.1, freq_mask_ratio: float = 0.1):
        self.mode = mode
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio

    def _mask(self, x: torch.Tensor, dim: int, ratio: float) -> torch.Tensor:
        if ratio <= 0.0:
            return x
        out = x.clone()
        size = x.size(dim)
        w = max(1, int(size * ratio))
        start = torch.randint(0, max(1, size - w + 1), (x.size(0),), device=x.device)
        for b in range(x.size(0)):
            if dim == 2:
                out[b, :, start[b]:start[b] + w, :] = 0
            else:
                out[b, :, :, start[b]:start[b] + w] = 0
        return out

    def __call__(self, x: torch.Tensor):
        v1 = x + self.noise_std * torch.randn_like(x)
        v2 = x + self.noise_std * torch.randn_like(x)
        if self.mode == "byol-a":
            # BYOL-A寄りに周波数マスクを強める
            v1 = self._mask(v1, dim=2, ratio=self.freq_mask_ratio)
            v2 = self._mask(v2, dim=2, ratio=self.freq_mask_ratio)
            v1 = self._mask(v1, dim=3, ratio=self.time_mask_ratio * 0.5)
            v2 = self._mask(v2, dim=3, ratio=self.time_mask_ratio * 0.5)
        else:
            # 通常BYOLは時間情報を残すため、時間マスクを弱くする
            v1 = self._mask(v1, dim=3, ratio=self.time_mask_ratio)
            v2 = self._mask(v2, dim=3, ratio=self.time_mask_ratio)
        return v1, v2
