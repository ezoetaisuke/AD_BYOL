import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import PurePath
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

from .datasets import create_loader
from .model_ae import Conv2dAE, recon_loss
from .utils import (
    ensure_dir, compute_roc_pr, plot_roc_pr, plot_confusion, plot_hist_by_class,
    extract_subclass_from_path, plot_hist_by_subclass, plot_hist_all_subclasses,
    save_spec_triplet_png,
)

from .calc_mahalanobis import (vectorize_residual, mahalanobis_distance,load_mahala_pack)

# =============================================================================
# この eval.py がやっていること
# -----------------------------------------------------------------------------
# 入力データ（OK/NG）を DataLoader で読み込み
#   → AutoEncoder で再構成 (reconstruction)
#   → 異常スコア算出（recon誤差 or Mahalanobis）
#   → 時間方向の集約（mean/max/topk_mean）
#   → 閾値決定（val OKのp99 / testでYouden / testでF1最大）
#   → CSV・混同行列・ROC/PR・ヒストグラム等を保存
#
# 重要：外部関数/クラス（create_loader, Conv2dAE, vectorize_residual 等）は別ファイル定義。
#       ここでは eval.py 内から読み取れる shape/意図をコメントします（中身は断定しません）。
# =============================================================================


# =============================================================================
# 処理フロー（ステップ番号付き / 上流→下流）
# -----------------------------------------------------------------------------
# 1. DataLoader作成
#    - cfg['data'] の各 *_glob を create_loader に渡して (x, y, paths) を得る
#    - train/val は OK(label=0) のみ、test は OK/NG を両方読む
# 2. モデル初期化
#    - cfg['model'] に従い Conv2dAE を構築（in_ch=1 前提）
# 3. 学習済み重み読み込み
#    - cfg['filenames'] の checkpoint から state_dict をロードし model.eval()
# 4. （必要なら）Mahalanobis統計量の読み込み
#    - cfg['scoring']['primary'] が mahalanobis 系のとき stats を load_mahala_pack
# 5. 推論→異常スコア算出
#    - x -> x_hat へ再構成
#    - recon_mse / recon_l1: residual(x-x_hat) を誤差マップとしてスコア化
#    - mahalanobis: residual を vectorize_residual でベクトル化し距離を算出
# 6. 時間方向の集約（1ファイル=1スコア）
#    - cfg['scoring']['aggregate_method'] : mean / max / topk_mean
# 7. 閾値決定
#    - cfg['threshold']['method'] により p99(val_ok) / Youden(test) / F1最大(test)
# 8. 評価指標算出
#    - y_true と scores から ROC/PR、混同行列、F1 などを計算
# 9. 保存物（成果物）
#    - scores.csv（path, y_true, score, y_pred）
#    - confusion_matrix.png / roc.png / pr.png / score_hist_all_subclasses.png
#    - （任意）再構成画像: cfg['eval']['save_recon_images'] が enable のとき保存
# =============================================================================

# =============================================================================
# shapeメモ（このファイルから推定できる範囲）
# -----------------------------------------------------------------------------
# x, x_hat:
#   - 推定: [B, 1, F, T] （B=バッチ, F=周波数bin, T=時間フレーム）
#   - 根拠: Conv2dAE(in_ch=1) を使い、可視化で x[i,0] を参照している
# residual / err_map:
#   - recon系: (x - x_hat) から作るため shape は x と同等
# md_per_t:
#   - Mahalanobis系: vectorize_residual の meta['B'], meta['T'] により [B, T] に復元
# file_scores:
#   - 1ファイル=1スコアに集約した結果なので shape は [B]
# =============================================================================


def _as_1d_float_list(x):
    # 各バッチで出た score を、最終的に Python の list[float] へ平坦化するユーティリティ。
    # - 入力: list/tuple/np.ndarray など
    # - 出力: 1次元の list[float]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    arr = np.asarray(x).reshape(-1)
    return [float(v) for v in arr.tolist()]


# -----------------------------------------------------------------------------
# _aggregate_time:
#   1サンプル内の「時間フレームごとのスコア（または誤差）」を、1つのスカラーへ集約します。
#   - 入力 err_map は 2D/3D/4D のいずれかを許容（docstring参照）。
#   - 出力は shape [B] の numpy 配列（=バッチ内の各サンプルの最終スコア）。
# 典型的な使い方：
#   - recon系: err_map=[B,1,F,T] を [B,T] に潰してから mean/max/topk_mean
#   - mahala系: per-frame md=[B,T] をそのまま集約
# 落とし穴：
#   - T（時間フレーム数）が極端に短い場合、topk_mean は k>=1 になるよう調整される
#   - 入力次元が想定外の場合は ValueError（データ前処理・モデル出力shapeに注意）
# -----------------------------------------------------------------------------
def _aggregate_time(err_map, method="topk_mean", topk_ratio=0.1):
    """Aggregate a time series of per-frame scores into a per-sample score.

    Acceptable shapes:
      - 4D: [B, C, F, T]  (e.g., residual energy map)
      - 3D: [B, F, T]
      - 2D: [B, T]        (already per-frame scores, e.g., Mahalanobis per frame)
    """
    err_map = np.asarray(err_map)
    if err_map.ndim == 4:
        per_t = err_map.mean(axis=2).squeeze(1)  # [B,T]
    elif err_map.ndim == 3:
        per_t = err_map.mean(axis=1)             # [B,T]
    elif err_map.ndim == 2:
        per_t = err_map                           # [B,T]
    else:
        raise ValueError(f"err_map must be 2D/3D/4D, got {err_map.shape}")

    if method == "mean":
        return per_t.mean(axis=1)
    if method == "max":
        return per_t.max(axis=1)
    if method == "topk_mean":
        k = max(1, int(np.ceil(per_t.shape[1] * float(topk_ratio))))
        part = np.partition(per_t, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    raise ValueError(f"Unknown aggregate method: {method}")

# -----------------------------------------------------------------------------
# calc_recon_mse:
#   再構成誤差（MSEベース）を異常スコアとして算出します。
#   - err_map = (x - x_hat)^2 を作り、時間方向Tに沿って _aggregate_time で [B] に集約
#   - 出力: file_scores（numpy, shape [B]）
# -----------------------------------------------------------------------------
def calc_recon_mse(x, x_hat, agg_method, topk_ratio):
    err_map = (x - x_hat).pow(2).detach().cpu().numpy()
    file_scores = _aggregate_time(err_map, method=agg_method, topk_ratio=topk_ratio)
    return file_scores

# -----------------------------------------------------------------------------
# calc_recon_l1:
#   再構成誤差（L1ベース）を異常スコアとして算出します。
#   - err_map = |x - x_hat| を作り、時間方向Tに沿って _aggregate_time で [B] に集約
#   - 出力: file_scores（numpy, shape [B]）
# -----------------------------------------------------------------------------
def calc_recon_l1(x, x_hat, agg_method, topk_ratio):
    err_map = (x - x_hat).abs().detach().cpu().numpy()
    file_scores = _aggregate_time(err_map, method=agg_method, topk_ratio=topk_ratio)
    return file_scores

# -----------------------------------------------------------------------------
# calc_mahalanobis_distance:
#   residual(x - x_hat) を「ベクトル化」して Mahalanobis 距離でスコア化します。
#   mahala_pack の想定キー（このファイルからの推定）：
#     - 'mu': 平均ベクトル（torch.Tensor）
#     - 'precision': 逆共分散（torch.Tensor）
#     - 'vectorize': vectorize_residual の mode 文字列
#   処理：
#     1) diff = x - x_hat
#     2) vectorize_residual(diff, mode=...) -> vecs と meta を取得
#        - meta には少なくとも 'B' と 'T' がある前提で view している
#     3) mahalanobis_distance(vecs, mu, precision) -> md（推定 shape [B*T]）
#     4) md を [B,T] に整形し、時間方向に集約して file_scores([B]) を返す
# 落とし穴：
#   - stats（mu/precision）が None の場合は RuntimeError
#   - vecs の次元が stats と一致しないと距離計算で例外になり得る（学習時設定と一致が必須）
# -----------------------------------------------------------------------------
def calc_mahalanobis_distance(device, mahala_pack, x, x_hat, agg_method, topk_ratio):

    md_mu = mahala_pack["mu"].to(device=device, dtype=torch.float32)
    md_precision = mahala_pack["precision"].to(device=device, dtype=torch.float32)
    md_vectorize = str(mahala_pack["vectorize"])

    if md_mu is None or md_precision is None:
        raise RuntimeError("Mahalanobis stats are not loaded (md_mu/md_precision is None).")
    diff = (x - x_hat)
    vecs, meta = vectorize_residual(diff, mode=md_vectorize)
    md = mahalanobis_distance(vecs, md_mu, md_precision, sqrt=True)  # [B*T]
    md_per_t = md.view(int(meta["B"]), int(meta["T"])).detach().cpu().numpy()  # [B,T]
    file_scores = _aggregate_time(md_per_t, method=agg_method, topk_ratio=topk_ratio) 

    return file_scores

# -----------------------------------------------------------------------------
# calc_selective_mahalanobis_distance:
#   2種類の Mahalanobis 統計量（source/target）でスコアを計算し、
#   その最小値を採用します（=どちらかの正常分布に近ければ低スコアになる設計）。
#   出力: file_scores（numpy, shape [B]）
# -----------------------------------------------------------------------------
def calc_selective_mahalanobis_distance(device, mahala_pack_source, mahala_pack_target, x, x_hat, agg_method, topk_ratio):
    
    file_scores_source = calc_mahalanobis_distance(device, mahala_pack_source, x, x_hat, agg_method, topk_ratio)
    file_scores_target = calc_mahalanobis_distance(device, mahala_pack_target, x, x_hat, agg_method, topk_ratio)

    file_scores = np.minimum(file_scores_source, file_scores_target)

    return file_scores


def run_eval(cfg):

    # ============================================================
    # 評価（eval）メインルーチン：
    #   入力データ（主に正常OK中心）からAEで再構成を行い、
    #   再構成誤差 or Mahalanobis距離を「異常スコア」として算出します。
    #   最終的に「1ファイル（1サンプル）= 1スコア」に集約し、閾値でOK/NG判定します。
    # ============================================================

    # -------------------------------------------------------------------------
    # run_eval(cfg) の入口と入出力
    # -------------------------------------------------------------------------
    # 入力:
    #   cfg: dict 形式の設定（YAML等からロードされる想定）
    #     - 必須に近いキー群（このファイルから参照しているもの）
    #       cfg['output_dir']
    #       cfg['data']['train_ok_glob'/'val_ok_glob'/'test_ok_glob'/'test_ng_glob']
    #       cfg['model']['type'/'bottleneck_dim']
    #       cfg['filenames'][checkpoint, scores_csv, confusion_matrix_png, roc_png, pr_png, ...]
    #       cfg['scoring']['primary'/'aggregate_method'/'topk_ratio']
    #       cfg['threshold']['method']
    #     - 条件付きキー
    #       primary が mahalanobis 系: cfg['mahala']['stats_path']
    #       primary が selective_mahala: cfg['selective_mahala']['save'][source/target_precision_path]
    #       再構成画像保存: cfg['eval']['save_recon_images']
    # 出力:
    #   scores_csv のパス（out_dir 配下）
    #   併せて out_dir に各種プロットPNG・（任意で）再構成画像が生成されます。
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ステップ0: 実行環境（デバイス）決定
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # ステップ1: 出力ディレクトリ決定
    #   - out_dir は「評価結果の保存先ルート」です。
    #   - パス関連のトラブル（存在しない/書き込み権限など）が起きやすいので運用時は要注意。
    # -------------------------------------------------------------------------
    out_dir = cfg["output_dir"]

    # -------------------------------------------------------------------------
    # ステップ2: DataLoader作成
    #   - create_loader は glob パターンに一致するデータを読みます（外部実装）。
    #   - ここでは shuffle=False にして、scores と paths の対応が崩れないようにしています。
    #   - cfg['data'][*] の glob は、ファイルが1件も無いと loader が空になり得ます。
    # -------------------------------------------------------------------------
    train_loader, _   = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False)
    val_loader, _     = create_loader(cfg["data"]["val_ok_glob"],   label=0, cfg=cfg, shuffle=False)
    test_ok_loader, _ = create_loader(cfg["data"]["test_ok_glob"],  label=0, cfg=cfg, shuffle=False)
    test_ng_loader, _ = create_loader(cfg["data"]["test_ng_glob"],  label=1, cfg=cfg, shuffle=False)

    # -------------------------------------------------------------------------
    # ステップ3: モデル構築
    #   - この eval.py は Conv2dAE 前提で書かれています（cfg['model']['type'] を assert）。
    #   - bottleneck_dim は AE の圧縮次元（小さいほど表現力が低く、異常が再構成されにくくなる傾向）。
    # -------------------------------------------------------------------------
    assert cfg["model"]["type"] == "conv2d_ae"
    model = Conv2dAE(in_ch=1, bottleneck_dim=cfg["model"]["bottleneck_dim"]).to(device)

    # -------------------------------------------------------------------------
    # ステップ4: ウォームアップforward（lazy初期化対策）
    #   - モデル内部で入力shapeに依存した初期化がある場合に備え、1バッチだけ forward しています。
    #   - 注意: train_loader が空だと StopIteration で落ちます（データglobの指定ミス等）。
    # -------------------------------------------------------------------------
    x0, _, _ = next(iter(train_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        _ = model(x0)
    model.to(device)

    # -------------------------------------------------------------------------
    # ステップ5: チェックポイント（学習済み重み）パス決定 → ロード
    #   - 優先: cfg['filenames']['checkpoint_best_full_path']（フルパス指定）
    #   - 無ければ: out_dir / cfg['filenames']['checkpoint_best']（相対パス指定）
    #   - 期待する checkpoint 形式:
    #       torch.load(...) で dict を得て、その中の state['model'] を load_state_dict する
    # -------------------------------------------------------------------------
    try:
        ckpt_path = cfg["filenames"]["checkpoint_best_full_path"]
    except:
        ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # 学習済み重みをロードし、model.eval() で推論モードにします（Dropout/BN等の挙動が固定）。
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # -------------------------------------------------------------------------
    # ステップ6: スコアリング設定（cfg['scoring']）
    #   - primary: スコア方式（recon_* / mahalanobis / selective_mahala）
    #   - aggregate_method: 時間方向Tの集約方式（mean/max/topk_mean）
    #   - topk_ratio: topk_mean の上位何割を平均するか
    #   補足:
    #   - recon_type / sigma2 は cfg から取得していますが、このファイル内では未使用です。
    #     （学習側/他モジュールとの整合性のために残っている可能性があります）
    # -------------------------------------------------------------------------
    primary    = cfg["scoring"]["primary"]
    agg_method = cfg["scoring"]["aggregate_method"]
    topk_ratio = cfg["scoring"]["topk_ratio"]
    recon_type = cfg["loss"]["recon_type"]
    sigma2     = cfg["loss"]["gaussian_nll_sigma2"]

    # -------------------------------------------------------------------------
    # ステップ7: （必要なら）Mahalanobis統計量のロード
    #   - primary が mahalanobis 系の場合のみ使用します。
    #   - stats_path は学習時に保存された統計量ファイルを指す想定です（外部実装）。
    # -------------------------------------------------------------------------
    if primary in ("mahala", "mahalanobis"):
        mahala_pack_path = cfg['mahala']['stats_path']
        mahala_pack = load_mahala_pack(mahala_pack_path, device)

    elif primary=="selective_mahala":
        src_path = cfg["selective_mahala"]["save"]["source_precision_path"]
        tgt_path = cfg["selective_mahala"]["save"]["target_precision_path"]
        mahala_pack_source = load_mahala_pack(src_path, device)
        mahala_pack_target = load_mahala_pack(tgt_path, device)

    # -------------------------------------------------------------------------
    # ステップ8: train_ok のスコア算出
    #   - ここで作る train_scores は、現状この関数内では閾値決定に使っていません。
    #   - 典型用途: 学習データ分布の確認やデバッグ（スコア分布の sanity check）
    # -------------------------------------------------------------------------
    train_scores = []
    with torch.no_grad():

        # train_loader からは (x, y, paths) が来ますが、ここでは y/paths は不要なので捨てています。
        for x, _, _ in tqdm(train_loader, desc="[train scoring]"):
            x = x.to(device)

            # 推論：x_hat は再構成特徴量（reconstruction）です。
            x_hat, _ = model(x)

            # --- スコア算出（primary別） ---
            # 以降で file_scores を作り、最終的に「1サンプル=1スコア」が確定します（file_scoresはshape [B]）。
            # recon系は residual(diff) = x - x_hat を誤差マップとして扱い、
            # Mahalanobis系は residualをベクトル化してフレームごとの距離を算出し、時間方向に集約します。
            if primary == "recon_mse":
                file_scores = calc_recon_mse(x, x_hat, agg_method, topk_ratio)
            elif primary == "recon_l1":
                file_scores = calc_recon_l1(x, x_hat, agg_method, topk_ratio)
            elif primary in ("mahala", "mahalanobis"):
                file_scores = calc_mahalanobis_distance(device, mahala_pack, x, x_hat, agg_method, topk_ratio)
            elif primary == "selective_mahala":
                file_scores = calc_selective_mahalanobis_distance(
                    device, mahala_pack_source, mahala_pack_target, 
                    x, x_hat, agg_method, topk_ratio
                    )
            else:
                raise ValueError(f"Unknown primary score: {primary}")

            train_scores.extend(_as_1d_float_list(file_scores))

    train_scores = np.asarray(train_scores, dtype=float)

    # -------------------------------------------------------------------------
    # ステップ9: val_ok のスコア算出
    #   - 閾値 method='p99_val_ok' の場合、val_ok の99パーセンタイルを閾値にします。
    #   - val_ok は「正常のみ」の前提なので、ラベルリークを避ける運用上の意図があります。
    # -------------------------------------------------------------------------
    v_scores = []
    with torch.no_grad():

        # val_loader からは (x, y, paths) が来ますが、ここでは y/paths は不要なので捨てています。
        for x, _, _ in tqdm(val_loader, desc="[val scoring]"):
            x = x.to(device)

            # 推論：x_hat は再構成特徴量（reconstruction）です。
            x_hat, _ = model(x)

            # --- スコア算出（primary別） ---
            # 以降で file_scores を作り、最終的に「1サンプル=1スコア」が確定します（file_scoresはshape [B]）。
            # recon系は residual(diff) = x - x_hat を誤差マップとして扱い、
            # Mahalanobis系は residualをベクトル化してフレームごとの距離を算出し、時間方向に集約します。
            if primary == "recon_mse":
                file_scores = calc_recon_mse(x, x_hat, agg_method, topk_ratio)
            elif primary == "recon_l1":
                file_scores = calc_recon_l1(x, x_hat, agg_method, topk_ratio)
            elif primary in ("mahala", "mahalanobis"):
                file_scores = calc_mahalanobis_distance(device, mahala_pack, x, x_hat, agg_method, topk_ratio)
            elif primary == "selective_mahala":
                file_scores = calc_selective_mahalanobis_distance(
                    device, mahala_pack_source, mahala_pack_target, 
                    x, x_hat, agg_method, topk_ratio
                    )
            else:
                raise ValueError(f"Unknown primary score: {primary}")

            v_scores.extend(_as_1d_float_list(file_scores))

    val_scores = np.asarray(v_scores, dtype=float)

    # -------------------------------------------------------------------------
    # ステップ10: test_ok / test_ng のスコア算出
    #   - scores: 推論して得た異常スコア（高いほど異常の想定）
    #   - y_true: 正解ラベル（loader側の y。OK=0, NG=1）
    #   - paths : 元ファイルパス（CSVや可視化の紐付けに使用）
    #   - 注意: test_ok_loader と test_ng_loader を順番に回すため、CSVもその順で並びます。
    # -------------------------------------------------------------------------
    scores, y_true, paths = [], [], []
    saved_count = 0 # 画像保存枚数カウント

    with torch.no_grad():
        for loader, true_label in [(test_ok_loader, 0), (test_ng_loader, 1)]:
            for (x, y, p) in tqdm(loader, desc="[test scoring]"):
                x = x.to(device)
                x_hat, _ = model(x)

                if primary == "recon_mse":
                    file_scores = calc_recon_mse(x, x_hat, agg_method, topk_ratio)
                elif primary == "recon_l1":
                    file_scores = calc_recon_l1(x, x_hat, agg_method, topk_ratio)
                elif primary in ("mahala", "mahalanobis"):
                    file_scores = calc_mahalanobis_distance(device, mahala_pack, x, x_hat, agg_method, topk_ratio)
                elif primary=="selective_mahala":
                    file_scores = calc_selective_mahalanobis_distance(
                        device, mahala_pack_source, mahala_pack_target, 
                        x, x_hat, agg_method, topk_ratio
                        )
                else:
                    raise ValueError(f"Unsupported primary: {primary}")

                scores.extend(_as_1d_float_list(file_scores))
                y_true.extend(y.detach().cpu().numpy().astype(int).tolist())
                paths.extend(list(p))

                # ---- 画像保存：この「サンプルのスコア」をタイトルに入れる ----
                vcfg = cfg['eval']['save_recon_images']
                if vcfg['enable']:
                    vis_dir = os.path.join(out_dir, vcfg['folder_name'])
                    os.makedirs(os.path.join(out_dir, vcfg['folder_name']), exist_ok=True)
                    for i in range(x.size(0)):
                        if (vcfg['max_items'] > 0) and (saved_count >= vcfg['max_items']):
                            break
                        X    = x[i, 0].detach().cpu().numpy()
                        Xhat = x_hat[i, 0].detach().cpu().numpy()
                        lbl_name  = PurePath(p[i]).parts[-2]
                        file_name = os.path.splitext(PurePath(p[i]).parts[-1])[0]
                        fig_store_dir_path = os.path.join(vis_dir, lbl_name)
                        os.makedirs(fig_store_dir_path, exist_ok=True)
                        out_png = os.path.join(fig_store_dir_path, f"{file_name}.png")
                        save_spec_triplet_png(
                            out_png, X, Xhat,
                            diff_mode=vcfg['diff_mode'], cmap=vcfg['cmap'], dpi=vcfg['dpi'],
                            score=float(file_scores[i]),
                        )
                        saved_count += 1

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    paths  = np.asarray(paths, dtype=object)

    # -------------------------------------------------------------------------
    # ステップ11: 閾値（thr）決定（cfg['threshold']['method']）
    #   - p99_val_ok:
    #       val_ok の99パーセンタイルを閾値にする（正常分布ベースで運用向き）
    #   - youden_test:
    #       test の ROC 曲線から Youden's J (tpr - fpr) 最大の点を採用
    #   - f1max_test:
    #       test のスコア候補を全探索し、F1が最大となる閾値を採用
    # 重要（運用上の注意）:
    #   - youden_test / f1max_test はテストデータを使って閾値を最適化するため、
    #     報告指標としては楽観的になり得ます（分割設計・再現性に注意）。
    # -------------------------------------------------------------------------
    method = cfg["threshold"]["method"]
    if method == "p99_val_ok":
        thr = float(np.percentile(val_scores, 99.0))
    elif method == "youden_test":
        from sklearn.metrics import roc_curve
        fpr, tpr, thr_list = roc_curve(y_true, scores)
        youden = tpr - fpr
        thr = float(thr_list[np.argmax(youden)])
    elif method == "f1max_test":
        thr_candidates = np.unique(scores)
        best_f1, best_thr = -1.0, None
        for t in thr_candidates:
            y_pred_tmp = (scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred_tmp)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        thr = float(best_thr)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    y_pred = (scores >= thr).astype(int)

    # -------------------------------------------------------------------------
    # ステップ12: CSV保存（scores.csv）
    #   - 1行=1サンプル（元ファイル）
    #   - score は連続値、y_pred は thr で2値化した推定ラベル（NG=1）
    #   - ensure_dir は「親ディレクトリ作成」をするユーティリティ（外部実装）
    # -------------------------------------------------------------------------
    scores_csv = os.path.join(out_dir, cfg["filenames"]["scores_csv"])
    ensure_dir(scores_csv)
    df = pd.DataFrame({"path": paths, "y_true": y_true, "score": scores, "y_pred": y_pred})
    df.to_csv(scores_csv, index=False)

    # -------------------------------------------------------------------------
    # ステップ13: 図表の保存
    #   - 混同行列（OK/NG）
    #   - ROC / PR（スコアが大きいほど異常、という前提で compute_roc_pr が計算する想定）
    # -------------------------------------------------------------------------
    cm_png = os.path.join(out_dir, cfg["filenames"]["confusion_matrix_png"])
    plot_confusion(cm_png, y_true, y_pred, labels=("OK","NG"))

    roc_pack, pr_pack = compute_roc_pr(y_true, scores)
    roc_png = os.path.join(out_dir, cfg["filenames"]["roc_png"])
    pr_png  = os.path.join(out_dir, cfg["filenames"]["pr_png"])
    plot_roc_pr(roc_png, pr_png, roc_pack, pr_pack)

    # -------------------------------------------------------------------------
    # ステップ14: サブクラス別のスコア分布可視化
    #   - extract_subclass_from_path は path からサブクラス名を抽出する外部関数。
    #   - ここでは全サンプル分の groups を作り、サブクラス別にヒストグラムを重ね描きします。
    # -------------------------------------------------------------------------
    groups = [extract_subclass_from_path(p) for p in paths]
    all_hist_sub = os.path.join(out_dir, cfg["filenames"]["score_hist_all_subclasses_png"])

    plot_hist_all_subclasses(all_hist_sub, scores, groups, title="Score Histogram (All Subclasses)")

    print(f"Saved: {scores_csv}")
    print(f"[Eval] primary={primary} thr={thr:.6f}  n={len(scores)}  ok={(y_true==0).sum()} ng={(y_true==1).sum()}")
    return scores_csv
