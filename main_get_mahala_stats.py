import os
import torch
import yaml
from tqdm import tqdm

from src.datasets import create_loader
from src.model_byol import BYOLModel
from src.calc_mahalanobis import vectorize_temporal_feature, cov_to_precision


@torch.no_grad()
def estimate_mahala_pack(model, loaders, device, eps=1.0e-6, use_pinv=True):
    """BYOL特徴から平均・共分散・精度行列を推定する。"""
    model.eval()
    sum_vec = None
    sum_outer = None
    n_total = 0
    last_meta = None

    for loader in loaders:
        for x, _, _ in tqdm(loader, desc="[estimate mahala]"):
            x = x.to(device)
            feat = model.encode(x)  # [B,D,T]
            vecs, meta = vectorize_temporal_feature(feat)
            last_meta = meta
            v = vecs.to(dtype=torch.float64)

            if sum_vec is None:
                d = v.size(1)
                sum_vec = torch.zeros(d, dtype=torch.float64, device=device)
                sum_outer = torch.zeros(d, d, dtype=torch.float64, device=device)

            sum_vec += v.sum(dim=0)
            sum_outer += v.t() @ v
            n_total += v.size(0)

    if n_total < 2:
        raise RuntimeError("Mahalanobis統計推定には2サンプル以上必要です。")

    mean = sum_vec / n_total
    cov = (sum_outer - n_total * torch.outer(mean, mean)) / (n_total - 1)
    precision = cov_to_precision(cov, eps=eps, use_pinv=use_pinv)

    return {
        "mean": mean.to("cpu", dtype=torch.float32),
        "cov": cov.to("cpu", dtype=torch.float32),
        "precision": precision.to("cpu", dtype=torch.float32),
        "n_total": int(n_total),
        "meta": last_meta,
        "feature_type": "byol_temporal",
    }


def run_get_mahala_stats(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    byol_cfg = cfg["model"]["byol"]
    n_mels = int(cfg["feature"]["logmel"]["n_mels"])
    model = BYOLModel(
        n_mels=n_mels,
        feat_dim=byol_cfg["feat_dim"],
        projector_hidden=byol_cfg["projector_hidden"],
        predictor_hidden=byol_cfg["predictor_hidden"],
        ema_decay=byol_cfg["ema_decay"],
        pretrained_path=byol_cfg.get("pretrained_path", ""),
    ).to(device)

    ckpt_path = cfg["filenames"].get("checkpoint_best_full_path", os.path.join(cfg["output_dir"], cfg["filenames"]["checkpoint_best"]))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    raw_globs = cfg["mahala"]["data"]["train_ok_glob"] + cfg["mahala"]["data"]["val_ok_glob"]
    ok_globs = [g for g in raw_globs if g and str(g).strip()]
    loaders = [create_loader(g, label=0, cfg=cfg, shuffle=False, drop_last=False)[0] for g in ok_globs]

    pack = estimate_mahala_pack(
        model=model,
        loaders=loaders,
        device=device,
        eps=float(cfg["mahala"]["eps"]),
        use_pinv=bool(cfg["mahala"]["use_pinv"]),
    )
    os.makedirs(os.path.dirname(cfg["mahala"]["stats_path"]), exist_ok=True)
    torch.save(pack, cfg["mahala"]["stats_path"])
    print(f"saved: {cfg['mahala']['stats_path']}")
    return cfg["mahala"]["stats_path"]


if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_get_mahala_stats(cfg)
