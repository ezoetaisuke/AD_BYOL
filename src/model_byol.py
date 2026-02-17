import copy
import importlib
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


class _OfficialBYOLAEncoder(nn.Module):
    """Wrap nttcslab/byol-a encoder and normalize output shape to [B,D,T]."""

    def __init__(self, n_mels: int, feat_dim: int, pretrained_path: str = ""):
        super().__init__()
        self.encoder = self._build_encoder(n_mels=n_mels, feat_dim=feat_dim)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.encoder.load_state_dict(state, strict=False)

    def _build_encoder(self, n_mels: int, feat_dim: int) -> nn.Module:
        try:
            models_mod = importlib.import_module("byol_a.models")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "byol_a package is required. Install BYOL-A from https://github.com/nttcslab/byol-a "
                "and make sure `byol_a` is importable."
            ) from e

        if not hasattr(models_mod, "AudioNTT2020"):
            raise AttributeError("byol_a.models.AudioNTT2020 is not available in the installed BYOL-A package")

        cls = getattr(models_mod, "AudioNTT2020")
        sig = inspect.signature(cls)
        kwargs = {}
        if "n_mels" in sig.parameters:
            kwargs["n_mels"] = n_mels
        if "d" in sig.parameters:
            kwargs["d"] = feat_dim
        elif "feat_dim" in sig.parameters:
            kwargs["feat_dim"] = feat_dim
        return cls(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if h.dim() == 2:
            h = h.unsqueeze(-1)  # [B,D] -> [B,D,1]
        elif h.dim() == 3 and h.shape[1] < h.shape[2]:
            h = h.transpose(1, 2)  # [B,T,D] -> [B,D,T]
        return F.normalize(h, dim=1)


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
    """BYOL-A official encoder + BYOL objective for anomaly-feature learning."""

    def __init__(
        self,
        n_mels: int,
        feat_dim: int = 128,
        projector_hidden: int = 256,
        predictor_hidden: int = 256,
        ema_decay: float = 0.99,
        pretrained_path: str = "",
    ):
        super().__init__()
        self.online_encoder = _OfficialBYOLAEncoder(n_mels=n_mels, feat_dim=feat_dim, pretrained_path=pretrained_path)
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
    def __init__(self, mode: str = "byol-a", noise_std: float = 0.02, time_mask_ratio: float = 0.1, freq_mask_ratio: float = 0.1):
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
        v1 = self._mask(v1, dim=2, ratio=self.freq_mask_ratio)
        v2 = self._mask(v2, dim=2, ratio=self.freq_mask_ratio)
        v1 = self._mask(v1, dim=3, ratio=self.time_mask_ratio)
        v2 = self._mask(v2, dim=3, ratio=self.time_mask_ratio)
        return v1, v2
