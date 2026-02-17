import os
import time
import torch
from torch import optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .datasets import create_loader
from .model_byol import BYOLModel, BYOLAugment
from .utils import set_seed, ensure_dir, save_metrics_csv, plot_learning_curve


def run_train(cfg):
    """正常データのみでBYOL/BYOL-Aを学習し、最良重みを保存する。"""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=True, drop_last=True)
    val_loader, _ = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)

    # ===== BYOLモデル初期化 =====
    byol_cfg = cfg["model"]["byol"]
    model = BYOLModel(
        in_ch=1,
        encoder_hidden=byol_cfg["encoder_hidden"],
        feat_dim=byol_cfg["feat_dim"],
        projector_hidden=byol_cfg["projector_hidden"],
        predictor_hidden=byol_cfg["predictor_hidden"],
        ema_decay=byol_cfg["ema_decay"],
    ).to(device)

    augment = BYOLAugment(
        mode=cfg["ssl"]["method"],
        noise_std=cfg["ssl"]["augment"]["noise_std"],
        time_mask_ratio=cfg["ssl"]["augment"]["time_mask_ratio"],
        freq_mask_ratio=cfg["ssl"]["augment"]["freq_mask_ratio"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["schedule"]["plateau"]["factor"],
        patience=cfg["schedule"]["plateau"]["patience"],
        min_lr=cfg["schedule"]["plateau"]["min_lr"],
    )

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    metrics_csv = os.path.join(out_dir, cfg["filenames"]["metrics_csv"])
    lc_png = os.path.join(out_dir, cfg["filenames"]["learning_curve_png"])
    ensure_dir(ckpt_path)
    ensure_dir(metrics_csv)
    ensure_dir(lc_png)

    patience = int(cfg["schedule"]["early_stopping"]["patience"])
    max_norm = float(cfg["train"]["grad_clip_norm"])
    epochs = int(cfg["train"]["epochs"])

    history = []
    best_val = float("inf")
    wait = 0

    autocast_kwargs = {"device_type": "cuda", "dtype": torch.float16, "enabled": use_amp}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ===== 学習フェーズ =====
        model.train()
        train_sum, train_n = 0.0, 0
        for x, _, _ in tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}"):
            x = x.to(device)
            x1, x2 = augment(x)

            optimizer.zero_grad(set_to_none=True)
            with autocast(**autocast_kwargs):
                loss = model.byol_loss(x1, x2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            model.update_target()

            bs = x.size(0)
            train_sum += float(loss.item()) * bs
            train_n += bs
        train_loss = train_sum / max(1, train_n)

        # ===== 検証フェーズ（同じBYOL損失で監視） =====
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, _, _ in tqdm(val_loader, desc=f"val epoch {epoch}/{epochs}"):
                x = x.to(device)
                x1, x2 = augment(x)
                with autocast(**autocast_kwargs):
                    vloss = model.byol_loss(x1, x2)
                bs = x.size(0)
                val_sum += float(vloss.item()) * bs
                val_n += bs
        val_loss = val_sum / max(1, val_n)

        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            wait = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            wait += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "time_sec": round(time.time() - t0, 2),
            }
        )
        save_metrics_csv(metrics_csv, history)
        plot_learning_curve(lc_png, history)

        print(f"[Epoch {epoch}] train_byol={train_loss:.6f} val_byol={val_loss:.6f} {'BEST ✔' if improved else ''}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training done. checkpoint: {ckpt_path}")
    return ckpt_path
