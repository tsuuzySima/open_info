#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4K PNGを入力として、32x32ブロックごとのモーションブラー強度マップを出力するスクリプト。
- 入力：画像ファイル（4K想定だが任意サイズ可）
- 出力：
    1) <out_prefix>_blocks.png        ... ブロック解像度（H/32 x W/32）のグレースケール強度画像
    2) <out_prefix>_full.png          ... 入力解像度に最近傍拡大した強度画像（ブロック境界を保つ）
    3) <out_prefix>_score.npy         ... 生のスコア行列（float32, shape=(H/32, W/32)）
オプション：--no-fft でFFTウェッジ確認をスキップ（勾配のみ）
依存：numpy, opencv-python (cv2)
使い方：
    python blur_strength_map.py input.png --out-prefix out/blur --block 32
"""
import argparse
import os
import sys
import numpy as np
import cv2

def pad_to_multiple(img, block=32):
    h, w = img.shape[:2]
    hpad = (block - (h % block)) % block
    wpad = (block - (w % block)) % block
    if hpad == 0 and wpad == 0:
        return img, (0,0,0,0)
    top = hpad // 2
    bottom = hpad - top
    left = wpad // 2
    right = wpad - left
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    return img_pad, (top,bottom,left,right)

def block_view2d(img2d, block=32):
    """2D配列を (by, bx, block, block) にビュー変換（コピー無し）"""
    h, w = img2d.shape
    by = h // block
    bx = w // block
    view = img2d.reshape(by, block, bx, block).swapaxes(1,2)
    return view  # shape = (by, bx, block, block)

def structure_tensor_sums(Ix, Iy, block=32):
    """各ブロックの構造テンソル要素の総和を返す"""
    v_xx = block_view2d(Ix*Ix, block).sum(axis=(-2,-1))
    v_yy = block_view2d(Iy*Iy, block).sum(axis=(-2,-1))
    v_xy = block_view2d(Ix*Iy, block).sum(axis=(-2,-1))
    return v_xx, v_yy, v_xy

def eigvals_and_theta_blur(Sxx, Syy, Sxy, eps=1e-12):
    """2x2対称行列の固有値・固有ベクトル角。blur方向はλminの固有ベクトル（θmax+90°）。"""
    T = Sxx + Syy
    D = np.sqrt((Sxx - Syy)**2 + 4.0*(Sxy**2) + eps)
    lmax = (T + D) * 0.5
    lmin = (T - D) * 0.5
    theta_max = 0.5 * np.arctan2(2.0*Sxy, (Sxx - Syy + eps))
    theta_blur = theta_max + (np.pi/2.0)
    theta_blur = (theta_blur + np.pi/2.0) % np.pi - np.pi/2.0  # [-pi/2, pi/2)
    return lmax, lmin, theta_blur

def fft_wedge_contrast(blocks, theta, wedge_width_deg=10.0, rmin=2, rmax=None, eps=1e-6):
    """
    各ブロック（32x32）に対して、推定ブラー角thetaの方向と直交方向の周波数エネルギーを比較し、
    Cf = W_perp / W_par を返す（大きいほどブラーらしい）。
    blocks: (by, bx, b, b) float32
    theta : (by, bx)      rad
    """
    by, bx, b, _ = blocks.shape
    blocks = blocks.astype(np.float32, copy=False)
    # ハン窓
    w1 = np.hanning(b).astype(np.float32)
    w2 = np.outer(w1, w1).astype(np.float32)
    win_blocks = blocks * w2  # ブロードキャスト

    # 2D FFT（ブロック毎）。最後の2軸のみ
    F = np.fft.fft2(win_blocks, axes=(-2,-1))
    F = np.fft.fftshift(F, axes=(-2,-1))
    L = np.log1p(np.abs(F).astype(np.float32))  # 対数スペクトル

    # 周波数座標（中心原点）
    x = (np.arange(b, dtype=np.float32) - (b/2.0))
    X, Y = np.meshgrid(x, x, indexing='xy')
    R = np.sqrt(X*X + Y*Y)
    if rmax is None:
        rmax = b/2.0
    valid = (R >= rmin) & (R <= rmax)
    Rnz = R + 1e-9
    cos_phi = (X / Rnz).astype(np.float32)
    sin_phi = (Y / Rnz).astype(np.float32)

    cos_phi4 = cos_phi[None, None, :, :]
    sin_phi4 = sin_phi[None, None, :, :]
    valid4 = valid[None, None, :, :]

    width = np.deg2rad(wedge_width_deg)
    cos_width = np.cos(width)

    ct = np.cos(theta).astype(np.float32)[:, :, None, None]
    st = np.sin(theta).astype(np.float32)[:, :, None, None]

    # 平行ウェッジ
    dcos_par = cos_phi4 * ct + sin_phi4 * st
    mask_par = (dcos_par >= cos_width) & valid4
    # 直交ウェッジ（θ+90°）
    ctp = -st  # cos(theta+pi/2)
    stp =  ct  # sin(theta+pi/2)
    dcos_perp = cos_phi4 * ctp + sin_phi4 * stp
    mask_perp = (dcos_perp >= cos_width) & valid4

    Wpar_den  = mask_par.sum(axis=(-2,-1)).astype(np.float32) + eps
    Wperp_den = mask_perp.sum(axis=(-2,-1)).astype(np.float32) + eps
    Wpar  = (L * mask_par).sum(axis=(-2,-1)) / Wpar_den
    Wperp = (L * mask_perp).sum(axis=(-2,-1)) / Wperp_den

    Cf = (Wperp + eps) / (Wpar + eps)
    return Cf.astype(np.float32)

def compute_blur_strength_map(img_bgr,
                              block=32,
                              gaussian_sigma=0.7,
                              e_percentile=20.0,
                              use_fft=True,
                              wedge_width_deg=10.0,
                              alpha=1.0,
                              beta=1.0):
    """
    入力BGR画像 -> ブロック毎のブラー強度（0..1）マップを返す。
    戻り値：S_norm (by, bx) float32, S_raw (by, bx) float32, meta(dict)
    """
    # 1) グレースケール（輝度）
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    gray = gray.astype(np.float32)

    # 2) パディング（32の倍数に）
    gray_pad, pads = pad_to_multiple(gray, block=block)
    h_pad, w_pad = gray_pad.shape
    by = h_pad // block
    bx = w_pad // block

    # 3) ノイズ低減（軽いガウシアン）
    if gaussian_sigma > 0:
        gray_pad = cv2.GaussianBlur(gray_pad, ksize=(0,0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)

    # 4) Sobel勾配
    Ix = cv2.Sobel(gray_pad, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(gray_pad, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    # 5) 構造テンソル要素のブロック総和
    Sxx, Syy, Sxy = structure_tensor_sums(Ix, Iy, block=block)

    # 6) 固有値とブラー角
    lmax, lmin, theta_b = eigvals_and_theta_blur(Sxx, Syy, Sxy)

    # 7) 異方性Rとテクスチャ量E
    eps = 1e-6
    R = (lmax / (lmin + eps)).astype(np.float32)
    E = (lmax + lmin).astype(np.float32)

    # 8) FFTウェッジによる確認（任意）
    if use_fft:
        # ブロックビューを作って一括FFT
        blocks = block_view2d(gray_pad, block=block).astype(np.float32)
        Cf = fft_wedge_contrast(blocks, theta_b.astype(np.float32), wedge_width_deg=wedge_width_deg)
    else:
        Cf = np.ones_like(R, dtype=np.float32)

    # 9) スコア統合 S = α log R + β log Cf
    R_clip = np.maximum(R, 1.0).astype(np.float32)
    Cf_clip = np.maximum(Cf, 1.0).astype(np.float32)
    S = (alpha * np.log(R_clip) + beta * np.log(Cf_clip)).astype(np.float32)

    # 10) 低テクスチャ棄却（Eが低いブロックは0に）
    tauE = np.percentile(E, e_percentile).astype(np.float32)
    lowtex = E < tauE
    S[lowtex] = 0.0

    # 11) 正規化（0..1）。上位パーセンタイルでスケーリングし飽和を軽減
    vmax = np.percentile(S, 99.0)
    if vmax <= 0:
        S_norm = np.zeros_like(S, dtype=np.float32)
    else:
        S_norm = np.clip(S / float(vmax), 0.0, 1.0).astype(np.float32)

    meta = {
        "pads": pads,
        "by": by, "bx": bx,
        "theta_blur_rad": theta_b.astype(np.float32),
        "anisotropy_R": R,
        "texture_E": E,
        "Cf": Cf,
        "tauE": float(tauE),
        "vmax_for_norm": float(vmax),
    }
    return S_norm, S, meta

def save_maps(S_norm, out_prefix, orig_size, block=32):
    """ブロック解像度画像と、入力解像度に最近傍拡大した画像を保存"""
    by, bx = S_norm.shape
    # 0..255に拡大
    S8 = np.round(S_norm * 255.0).astype(np.uint8)
    # ブロック解像度のPNG
    path_blocks = f"{out_prefix}_blocks.png"
    cv2.imwrite(path_blocks, S8)

    # 入力解像度に最近傍拡大（ブロック境界を維持）
    H, W = orig_size
    full = cv2.resize(S8, (W, H), interpolation=cv2.INTER_NEAREST)
    path_full = f"{out_prefix}_full.png"
    cv2.imwrite(path_full, full)

    return path_blocks, path_full

def main():
    ap = argparse.ArgumentParser(description="32x32ブロックごとのモーションブラー強度マップを生成")
    ap.add_argument("input", help="入力画像（PNG/JPEG等）")
    ap.add_argument("--out-prefix", default="blur_strength", help="出力ファイル接頭辞")
    ap.add_argument("--block", type=int, default=32, help="ブロックサイズ（既定: 32）")
    ap.add_argument("--sigma", type=float, default=0.7, help="ガウシアンぼかしσ（ノイズ低減, 0で無効）")
    ap.add_argument("--wedge-width", type=float, default=10.0, help="FFTウェッジ幅（度）")
    ap.add_argument("--no-fft", action="store_true", help="FFTウェッジ確認をスキップ（勾配のみ）")
    ap.add_argument("--alpha", type=float, default=1.0, help="log(R) の重み")
    ap.add_argument("--beta", type=float, default=1.0, help="log(Cf) の重み（--no-fft時は無視）")
    ap.add_argument("--e-percentile", type=float, default=20.0, help="低テクスチャ棄却のパーセンタイル（%）")
    args = ap.parse_args()

    # 画像読み込み
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] 画像が読み込めませんでした: {args.input}", file=sys.stderr)
        sys.exit(1)
    H, W = img.shape[:2]

    # 計算
    S_norm, S_raw, meta = compute_blur_strength_map(
        img_bgr=img,
        block=args.block,
        gaussian_sigma=args.sigma,
        e_percentile=args.e_percentile,
        use_fft=(not args.no_fft),
        wedge_width_deg=args.wedge_width,
        alpha=args.alpha,
        beta=args.beta,
    )

    # 保存
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    path_blocks, path_full = save_maps(S_norm, args.out_prefix, orig_size=(H, W), block=args.block)

    # スコア行列も保存
    np.save(f"{args.out_prefix}_score.npy", S_raw.astype(np.float32))

    # 角度マップなどは必要に応じて保存可能（例：np.saveで）
    # ここでは出力パスのみ表示
    print("[INFO] 出力:")
    print(f"  ブロック解像度: {path_blocks}")
    print(f"  入力解像度    : {path_full}")
    print(f"  スコアnpy     : {args.out_prefix}_score.npy")
    print(f"  低テクスチャ閾値 tauE = {meta['tauE']:.3e}, 正規化上限 p99 = {meta['vmax_for_norm']:.3e}")
    if not args.no_fft:
        print("  FFTウェッジ確認: 有効")
    else:
        print("  FFTウェッジ確認: 無効（勾配のみ）")

if __name__ == "__main__":
    main()
