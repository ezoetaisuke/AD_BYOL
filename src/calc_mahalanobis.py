import torch


def load_mahala_pack(path: str, device: torch.device) -> dict:
    pack = torch.load(path, map_location="cpu")
    precision = pack.get("precision", None)
    mean = pack.get("mean", pack.get("mu", None))
    if precision is None or mean is None:
        raise KeyError(f"Mahalanobis pack missing keys. need precision+mean: {path}")
    pack["precision"] = precision.to(device=device, dtype=torch.float32)
    pack["mean"] = mean.to(device=device, dtype=torch.float32)
    return pack


def vectorize_temporal_feature(feat: torch.Tensor):
    """[B,D,T] 形式の特徴を [B*T,D] に変換する。"""
    if feat.dim() != 3:
        raise ValueError(f"feat must be 3D [B,D,T], got {tuple(feat.shape)}")
    B, D, T = feat.shape
    vecs = feat.permute(0, 2, 1).contiguous().view(B * T, D)
    meta = {"B": B, "D": D, "T": T}
    return vecs, meta


def cov_to_precision(cov: torch.Tensor, eps: float = 1.0e-6, use_pinv: bool = True):
    if cov.dim() != 2 or cov.size(0) != cov.size(1):
        raise ValueError(f"cov must be square 2D, got {tuple(cov.shape)}")
    D = cov.size(0)
    eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
    cov_reg = cov + float(eps) * eye
    prec = torch.linalg.pinv(cov_reg) if use_pinv else torch.linalg.inv(cov_reg)
    return prec.to(dtype=torch.float32)


def mahalanobis_distance(x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor, sqrt: bool = True):
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N,D], got {tuple(x.shape)}")
    xc = x.to(dtype=torch.float32) - mu.to(dtype=torch.float32)
    left = xc @ precision.to(dtype=torch.float32)
    d2 = (left * xc).sum(dim=1).clamp_min(0.0)
    return torch.sqrt(d2) if sqrt else d2
