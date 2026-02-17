import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # ★ 追加：Seabornで見た目改善
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# 乱数固定：再現性を高める（完全決定論は性能低下の場合がある）
def set_seed(seed: int):
    """
    乱数シードを一括固定し、実験の再現性を確保する。
    
    Args:
        seed (int): 固定するシード値
    
    Note:
        - 完全な決定論的挙動（deterministic）を強制すると、畳み込み演算などの
          アルゴリズムが制限され、処理速度が低下する場合がある。
        - GPUを使用する場合、cuDNNのベンチマーク機能をオフにすることで
          入力サイズが変わっても同じアルゴリズムが選ばれるようにする。
    """

    # Python標準の乱数固定
    random.seed(seed)

    # Numpyの乱数固定
    np.random.seed(seed)

    # PyTorch CPU/GPUの乱数固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)    # マルチGPUの場合

    # cuDNNの挙動を固定（再現性重視の設定）
    # 決定論的アルゴリズムのみを使用するように強制
    torch.backends.cudnn.deterministic = True

    # 最適なアルゴリズムを動的に探す機能をオフ（入力サイズ固定なら再現性に寄与）
    torch.backends.cudnn.benchmark = False

# path から親ディレクトリ部分だけを取り出す（例: "a/b/c.csv" -> "a/b"）
# 親ディレクトリが存在しない場合のみ作成し、存在する場合は何もしない
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 学習履歴をCSVに保存
def save_metrics_csv(out_csv: str, history: list):
    """
    学習履歴 `history` を Pandas DataFrame に変換し、CSV として保存する。

    Parameters
    ----------
    out_csv : str
        出力CSVファイルパス（例: "runs/exp01/metrics.csv"）。
        事前に `ensure_dir(out_csv)` を呼ぶことで、親フォルダが無ければ作成される。
    history : list
        学習履歴。一般に `dict` のリストを想定（例: [{"epoch":1,"train_loss":...,"val_loss":...}, ...]）。
        DataFrame化により、dict のキーが列名になる。

    処理の流れ（ロジックは現状のまま）
    --------------------------------
    1) `history` を DataFrame 化（行＝エポックなどの記録、列＝指標名）
    2) 出力先 `out_csv` の親ディレクトリを作成（必要なら）
    3) CSV へ保存（index=False で行番号列は出さない）
    """
    # list[dict] を DataFrame に変換（キーが列名になる）
    df = pd.DataFrame(history)

    # 保存先CSVの親ディレクトリを作成（無ければ作る / あれば何もしない）
    ensure_dir(out_csv)

    # CSV書き出し：index=False で DataFrame のインデックス列を保存しない
    df.to_csv(out_csv, index=False)

# 学習曲線（ELBO）を保存
def plot_learning_curve(out_png: str, history: list):
    """
    学習履歴 `history` から「epoch vs loss（train/val）」の学習曲線を作成し、PNGとして保存する。

    Parameters
    ----------
    out_png : str
        出力PNGファイルパス（例: "runs/exp01/learning_curve.png"）。
        先に `ensure_dir(out_png)` を呼び、親ディレクトリが無ければ作成してから保存する。
    history : list
        学習履歴。要素は dict を想定し、最低限以下のキーを持つ前提:
          - "epoch"      : エポック番号（x軸）
          - "train_loss" : 訓練損失（y軸）
          - "val_loss"   : 検証損失（y軸）
        例: [{"epoch":1,"train_loss":..., "val_loss":...}, ...]

    Notes
    -----
    - 可視化は `matplotlib.pyplot` のステートフルAPI（plt.*）で描画しているため、
      `fig` を明示的に close してメモリリーク/図の積み上がりを防いでいる。
    - y軸ラベルは "Loss (lower is better)" としているため、指標が損失である想定。
    """

    # 保存先PNGの親ディレクトリを作成（無ければ作る / あれば何もしない）
    ensure_dir(out_png)

    # history からプロット用の系列を抽出
    # - epochs : x軸（エポック番号）
    # - tr_elbo/va_elbo : y軸（train/val の損失系列）
    epochs = [h["epoch"] for h in history]
    tr_elbo = [h["train_loss"] for h in history]
    va_elbo = [h["val_loss"] for h in history]

    # Figure を作成（plt.* で描画するが、最後に close するため参照を保持する）
    fig = plt.figure(figsize=(6,4))
    
    # 学習曲線（train/val）を同一グラフ上に描画
    plt.plot(epochs, tr_elbo, label="train loss")
    plt.plot(epochs, va_elbo, label="val loss")

    # 軸ラベル・凡例・グリッド設定（可読性を確保）
    plt.xlabel("Epoch"); plt.ylabel("Loss (lower is better)")
    plt.legend(); plt.grid(True)

    # 余白を詰めて保存（ラベル切れ防止）
    plt.tight_layout()

    # PNGとして保存（dpi=150 で適度な解像度）
    plt.savefig(out_png, dpi=150)

    # Figure を閉じてリソース解放（ループで多枚数保存するケースを想定）
    plt.close(fig)

# ROC/PR 計算
def compute_roc_pr(y_true, scores):
    fpr, tpr, thr_roc = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, thr_pr = precision_recall_curve(y_true, scores)
    pr_auc = auc(rec, prec)
    return (fpr, tpr, thr_roc, roc_auc), (prec, rec, thr_pr, pr_auc)

# ROC/PR 図
def plot_roc_pr(out_roc, out_pr, roc_pack, pr_pack):
    ensure_dir(out_roc); ensure_dir(out_pr)
    fpr, tpr, _, roc_auc = roc_pack
    prec, rec, _, pr_auc = pr_pack
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_roc, dpi=150); plt.close(fig1)

    fig2 = plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_pr, dpi=150); plt.close(fig2)

# ===============================
# ヒストグラム（Seabornスタイルに刷新）
# ===============================
def _shared_bins_from_arrays(*arrays, nbins: int = 20):
    """
    配列群の全体最小/最大から共有bin（np.linspace）を作る。
    すべてNaN or 同値の場合は固定本数で返す。
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if len(a) > 0]) if len(arrays) > 0 else np.asarray([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 20
    vmin, vmax = np.min(vals), np.max(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 10
    return np.linspace(vmin, vmax, nbins + 1)

# OK/NG 2色ヒスト（Seaborn版）
def plot_hist_by_class(out_png, scores_ok, scores_ng, title: str = "Anomaly Score Histogram",
                       xlabel: str = "Anomaly Score", ylabel: str = "Count",
                       nbins: int = 20):
    """
    OK(青)/NG(赤)の2色ヒストを Seaborn histplot で描画。
    - kde=True（分布のなめらか曲線）
    - stat='count'（縦軸は件数）
    - element='step'（縁取りステップ表示）
    """
    ensure_dir(out_png)
    bins = _shared_bins_from_arrays(scores_ok, scores_ng, nbins=nbins)

    df = pd.DataFrame({
        "score": np.concatenate([np.asarray(scores_ok, dtype=float), np.asarray(scores_ng, dtype=float)]),
        "label": (["OK"] * len(scores_ok)) + (["NG"] * len(scores_ng))
    })
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(
        data=df, x="score", hue="label",
        bins=bins, kde=True, stat="count", element="step", alpha=0.5, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# 混同行列
def plot_confusion(out_png, y_true, y_pred, labels=("OK","NG")):
    ensure_dir(out_png)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black", fontsize=12)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# 親フォルダ名をサブクラス名として抽出
def extract_subclass_from_path(path: str) -> str:
    base = os.path.basename(os.path.dirname(path))
    return base

# サブクラス別（OKのみ or NGのみなど）ヒスト（Seaborn版）
def plot_hist_by_subclass(out_png: str, scores: np.ndarray, groups: list, title: str,
                          xlabel: str = "Anomaly Score", ylabel: str = "Count",
                          nbins: int = 20):
    """
    同一集合（OKのみ、NGのみなど）内でサブクラスごとに色分けヒスト。
    KDE付き、共有bin、countスケール、step表示。
    """
    ensure_dir(out_png)
    scores = np.asarray(list(scores), dtype=float)
    uniq = sorted(list(set(groups)))
    bins = _shared_bins_from_arrays(scores, nbins=nbins)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab20", n_colors=max(20, len(uniq)))

    for i, g in enumerate(uniq):
        s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
        if s.size == 0:
            continue
        sns.histplot(s,
                     alpha=0.6, label=g, ax=ax, stat='count', bins=bins, element='step')

    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fd_bin_edges(values: np.ndarray, min_bins: int = 12, max_bins: int = 24):
    """
    Freedman–Diaconisでビン幅を決めて、ビン数を[min_bins, max_bins]にクランプする。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0, 1, min_bins + 1)

    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        return np.linspace(vmin - 0.5, vmax + 0.5, min_bins + 1)

    # IQR
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float(q75 - q25)

    # FDのビン幅 h
    if iqr > 0.0:
        h = 2.0 * iqr / np.cbrt(v.size)
    else:
        # IQR=0 の退避（Scottの幅 or 固定幅）
        sd = float(np.std(v))
        h = 3.5 * sd / np.cbrt(v.size) if sd > 0 else (vmax - vmin) / max(min_bins, 1)

    if h <= 0:
        nbins = max(min_bins, 1)
    else:
        nbins = int(np.ceil((vmax - vmin) / h))
        nbins = int(np.clip(nbins, min_bins, max_bins))

    # 等間隔の境界に整形
    return np.linspace(vmin, vmax, nbins + 1)


def _shared_bins_fd_clamped(arrays, min_bins: int = 12, max_bins: int = 24):
    """
    複数配列（OK/NG/サブクラスなど）を結合して
    FD＋クランプで “共有bin” 境界を返す。
    """
    concat = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays if len(a) > 0], axis=0)
    return _fd_bin_edges(concat, min_bins=min_bins, max_bins=max_bins)

def save_spec_triplet_png(
    out_png: str,
    X: np.ndarray,        # [F, T] エンコーダ入力（特徴量）
    Xhat: np.ndarray,     # [F, T] デコーダ出力（再構成特徴量）
    diff_mode: str = "abs",   # "abs" or "signed"
    cmap: str = "magma",
    dpi: int = 150,
    score:float = None
):
    """
    入力, 再構成, 差分 を横一列に並べて保存する。
    - 入力/再構成は同じ vmin/vmax を共有して比較をしやすく。
    - 差分は diff_mode に応じて |X-Xhat| か (X-Xhat) を描画。
    """
    ensure_dir(out_png)

    X = np.asarray(X, dtype=float)
    Xhat = np.asarray(Xhat, dtype=float)

    # 入力・再構成は同一カラースケールで比較
    vmin = float(np.nanmin([X.min(), Xhat.min()]))
    vmax = float(np.nanmax([X.max(), Xhat.max()]))

    if diff_mode == "signed":
        D = X - Xhat
        d_absmax = float(np.max(np.abs(D)))
        d_vmin, d_vmax = -d_absmax, d_absmax
        diff_cmap = "coolwarm"
    else:
        D = np.abs(X - Xhat)
        d_vmin, d_vmax = float(D.min()), float(D.max())
        diff_cmap = cmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(X, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Input")
    axes[1].imshow(Xhat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Recon")
    axes[2].imshow(D, aspect="auto", origin="lower", cmap=diff_cmap, vmin=d_vmin, vmax=d_vmax)
    axes[2].set_title("Diff")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    if score is not None:
        fig.suptitle(f"score={score:.6f}")

    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)

# ★OK/NGすべてのサブクラスを1枚に統合して色分け（Seaborn版）
def plot_hist_all_subclasses(
        out_png: str, 
        scores: np.ndarray, 
        groups: list, 
        title: str,
        xlabel: str = "Anomaly Score", 
        ylabel: str = "Count",
        bins="fd_clamped",
        min_bins: int = 12,
        max_bins: int = 24,
    ):

    """
    OK/NGを含む全サブクラスを1枚で色分け表示。
    ラベルは '... (OK)' / '... (NG)' のようにタグ付け。
    KDE付き、共有bin、countスケール、step表示。
    """
    ensure_dir(out_png)
    scores = np.asarray(list(scores), dtype=float)
    uniq = sorted(list(set(groups)))

    def decorate(name: str) -> str:
        tag = "OK" if "ok" in name.lower() else ("NG" if "ng" in name.lower() else "")
        return f"{name} ({tag})" if tag else name

    if isinstance(bins, str) and bins == "fd_clamped":
        bins_ = _shared_bins_fd_clamped([scores], min_bins, max_bins)
    else:
        bins_ = bins

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, g in enumerate(uniq):
        s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
        mu = float(np.mean(s))
        sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
        s_max = max(s)
        s_min = min(s)
        if s.size == 0:
            continue
        ax.hist(
            s,
            bins=20,
            alpha=0.8,
            histtype="stepfilled",
            # edgecolor="k",
            # linewidth=0.5,
            label=f"{g} (μ={mu:.2f}, σ={sd:.2f}, max={s_max:.2f}, min={s_min:.2f}, n={s.size})",
        )
        # ax.vlines(s, ymin=0, ymax=max(1, int(0.05 * s.size)), colors="k", alpha=0.35, linewidth=0.8)

    # ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

