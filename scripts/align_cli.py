#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mmalign import io as mmio
from mmalign import align as mmalign


def load_ts(path: str, name: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return mmio.load_csv_timeseries(path, name)
    elif ext in {'.wav', '.mp3', '.flac'}:
        return mmio.load_audio_timeseries(path, name)
    elif ext in {'.mp4', '.mov', '.avi'}:
        return mmio.load_video_timeseries(path, name)
    elif ext in {'.srt', '.txt'}:
        return mmio.load_text_srt_timeseries(path, name)
    else:
        raise ValueError(f'Unsupported file: {path}')


def main():
    parser = argparse.ArgumentParser(description='对齐两个流并导出对齐叠加CSV')
    parser.add_argument('--ref', required=True, help='参考流文件（CSV/WAV/MP4/SRT 等）')
    parser.add_argument('--tgt', required=True, help='目标流文件（CSV/WAV/MP4/SRT 等）')
    parser.add_argument('--method', choices=['resample', 'xcorr', 'dtw'], default='xcorr')
    parser.add_argument('--rate', type=float, default=50.0, help='目标采样率Hz')
    parser.add_argument('--max_lag', type=float, default=5.0, help='互相关最大时滞(s)')
    parser.add_argument('--out', required=True, help='输出CSV路径')
    args = parser.parse_args()

    sref = load_ts(args.ref, 'ref')
    stgt = load_ts(args.tgt, 'tgt')

    if args.method == 'resample':
        ref_r = mmalign.resample_to_rate(sref, target_rate_hz=args.rate)
        tgt_r = mmalign.resample_to_rate(stgt, target_rate_hz=args.rate)
        overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
    elif args.method == 'xcorr':
        lag = mmalign.estimate_lag_by_xcorr(sref, stgt, max_lag_seconds=args.max_lag)
        stgt_shift = mmalign.shift_series_in_time(stgt, shift_seconds=-lag)
        ref_r = mmalign.resample_to_rate(sref, target_rate_hz=args.rate)
        tgt_r = mmalign.resample_to_rate(stgt_shift, target_rate_hz=args.rate)
        overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
        print(f"Estimated lag: {lag:.3f}s")
    else:  # dtw
        ref_r = mmalign.resample_to_rate(sref, target_rate_hz=args.rate)
        tgt_r = mmalign.resample_to_rate(stgt, target_rate_hz=args.rate)
        mapping, dist = mmalign.dtw_mapping(ref_r, tgt_r)
        overlay_df = mmalign.apply_mapping_and_overlay(ref_r, tgt_r, mapping)
        print(f"DTW distance: {dist:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    overlay_df.to_csv(args.out, index=False)
    print(f"✅ saved aligned overlay -> {args.out}  (rows={len(overlay_df)})")


if __name__ == '__main__':
    main()


