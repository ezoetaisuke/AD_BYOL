import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .datasets import create_loader
from .model_byol import BYOLModel
from .calc_mahalanobis import (
    cov_to_precision,
    load_mahala_pack,
    mahalanobis_distance,
    vectorize_temporal_feature,
)
from .utils import (
    ensure_dir,
    compute_roc_pr,
    plot_roc_pr,
    plot_confusion,
    extract_subclass_from_path,
    plot_hist_all_subclasses,
)


def _aggregate_time(per_t: np.ndarray, method: str, topk_ratio: float):
    if method == "mean":
        return per_t.mean(axis=1)
    if method == "max":
        return per_t.max(axis=1)
    if method == "topk_mean":
        k = max(1, int(per_t.shape[1] * topk_ratio))
        idx = np.argpartition(per_t, kth=per_t.shape[1] - k, axis=1)[:, -k:]
        return np.take_along_axis(per_t, idx, axis=1).mean(axis=1)
    raise ValueError(f"unknown aggregate method: {method}")


@torch.no_grad()
def _score_loader(model, loader, pack, device, agg_method, topk_ratio):
    all_scores, all_labels, all_paths = [], [], []
    model.eval()

    for x, y, paths in tqdm(loader, desc="[scoring]"):
        x = x.to(device)
        feat = model.encode(x)  # [B,D,T]
        vecs, meta = vectorize_temporal_feature(feat)
        md = mahalanobis_distance(vecs, pack["mean"], pack["precision"], sqrt=True)
        md_bt = md.view(meta["B"], meta["T"]).cpu().numpy()
        file_scores = _aggregate_time(md_bt, method=agg_method, topk_ratio=topk_ratio)

        all_scores.extend(file_scores.tolist())
        all_labels.extend(y.numpy().astype(int).tolist())
        all_paths.extend(list(paths))

    return np.asarray(all_scores, dtype=float), np.asarray(all_labels, dtype=int), all_paths


@torch.no_grad()
def _infer_feature_dim(model, loader, device):
    for x, _, _ in loader:
        feat = model.encode(x.to(device))
        _, D, _ = feat.shape
        return int(D)
    raise ValueError("loader is empty; cannot infer feature dimension")


@torch.no_grad()
def _estimate_mahala_pack(model, loader, device, eps: float, use_pinv: bool):
    vecs_all = []
    for x, _, _ in tqdm(loader, desc="[fit-mahala]"):
        feat = model.encode(x.to(device))
        vecs, _ = vectorize_temporal_feature(feat)
        vecs_all.append(vecs.to(dtype=torch.float32))

    if not vecs_all:
        raise ValueError("cannot estimate Mahalanobis stats from an empty loader")

    x = torch.cat(vecs_all, dim=0)
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(1, (xc.size(0) - 1))
    precision = cov_to_precision(cov, eps=eps, use_pinv=use_pinv)
    return {
        "mean": mean.to(device=device, dtype=torch.float32),
        "precision": precision.to(device=device, dtype=torch.float32),
    }


def run_eval(cfg):
    """BYOL特徴→Mahalanobis距離で異常判定する評価ルーチン。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output_dir"]

    # ===== データ準備 =====
    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)
    val_loader, _ = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)
    test_ok_loader, _ = create_loader(cfg["data"]["test_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)
    test_ng_loader, _ = create_loader(cfg["data"]["test_ng_glob"], label=1, cfg=cfg, shuffle=False, drop_last=False)

    # ===== モデルロード =====
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

    ckpt_path = cfg["filenames"].get("checkpoint_best_full_path", os.path.join(out_dir, cfg["filenames"]["checkpoint_best"]))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # ===== Mahalanobis統計量 =====
    stats_path = cfg["mahala"].get("stats_path", "")
    if not stats_path:
        stats_path = os.path.join(out_dir, cfg["filenames"]["mahala_stats_pt"])
    pack = load_mahala_pack(stats_path, device)

    feat_dim = _infer_feature_dim(model, val_loader, device)
    pack_dim = int(pack["mean"].numel())
    if pack_dim != feat_dim:
        eps = float(cfg["mahala"].get("eps", 1.0e-6))
        use_pinv = bool(cfg["mahala"].get("use_pinv", True))
        print(
            f"[warn] Mahalanobis stats dim mismatch: model={feat_dim}, stats={pack_dim} ({stats_path}). "
            "Re-estimating stats from train_ok loader."
        )
        pack = _estimate_mahala_pack(model, train_loader, device, eps=eps, use_pinv=use_pinv)

    agg_method = cfg["scoring"]["aggregate_method"]
    topk_ratio = float(cfg["scoring"]["topk_ratio"])

    # ===== スコア算出 =====
    val_scores, _, _ = _score_loader(model, val_loader, pack, device, agg_method, topk_ratio)
    ok_scores, ok_labels, ok_paths = _score_loader(model, test_ok_loader, pack, device, agg_method, topk_ratio)
    ng_scores, ng_labels, ng_paths = _score_loader(model, test_ng_loader, pack, device, agg_method, topk_ratio)

    # ===== 閾値決定（正常検証データのp99） =====
    threshold = float(np.percentile(val_scores, 99))

    y_true = np.concatenate([ok_labels, ng_labels])
    y_score = np.concatenate([ok_scores, ng_scores])
    y_pred = (y_score >= threshold).astype(int)
    paths = ok_paths + ng_paths

    # ===== 結果保存 =====
    scores_csv = os.path.join(out_dir, cfg["filenames"]["scores_csv"])
    ensure_dir(scores_csv)
    pd.DataFrame({
        "path": paths,
        "y_true": y_true,
        "score": y_score,
        "y_pred": y_pred,
        "threshold": threshold,
    }).to_csv(scores_csv, index=False)

    roc_pack, pr_pack = compute_roc_pr(y_true, y_score)
    plot_roc_pr(os.path.join(out_dir, cfg["filenames"]["roc_png"]), os.path.join(out_dir, cfg["filenames"]["pr_png"]), roc_pack, pr_pack)
    plot_confusion(os.path.join(out_dir, cfg["filenames"]["confusion_matrix_png"]), y_true, y_pred)
    subclass_groups = [extract_subclass_from_path(p) for p in paths]
    hist_path = os.path.join(
        out_dir,
        cfg["filenames"].get("score_hist_all_subclasses_png", cfg["filenames"]["score_hist_png"]),
    )
    plot_hist_all_subclasses(
        hist_path,
        y_score,
        subclass_groups,
        title="Score Histogram (All Subclasses)",
    )

    print(f"Eval done. threshold={threshold:.6f}, roc_auc={roc_pack[3]:.4f}")
    return {
        "threshold": threshold,
        "roc_auc": float(roc_pack[3]),
        "pr_auc": float(pr_pack[3]),
    }
