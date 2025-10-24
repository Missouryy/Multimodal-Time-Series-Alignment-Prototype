#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
from mmalign import io as mmio


def save_ts_to_csv(ts, out_csv: str):
    df = ts.to_dataframe()
    df.to_csv(out_csv, index=False)
    print(f"✅ saved CSV -> {out_csv}  (rows={len(df)})")


def main():
    parser = argparse.ArgumentParser(description='预处理多模态文件并导出标准 CSV (time,v*)')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--audio', help='音频文件: wav/mp3/flac')
    g.add_argument('--video', help='视频文件: mp4/mov/avi')
    g.add_argument('--text', help='文本: srt 或 含 time 列的 csv/tsv')
    g.add_argument('--sensor', help='传感器 CSV: 必须包含 time/timestamp 列')
    parser.add_argument('--name', help='流名称', default=None)
    parser.add_argument('--out', help='输出 CSV 路径', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    if args.sensor:
        ts = mmio.load_csv_timeseries(args.sensor, name=args.name or os.path.splitext(os.path.basename(args.sensor))[0])
        save_ts_to_csv(ts, args.out)
        return

    if args.audio:
        ts = mmio.load_audio_timeseries(args.audio, name=args.name or os.path.splitext(os.path.basename(args.audio))[0])
        save_ts_to_csv(ts, args.out)
        return

    if args.video:
        ts = mmio.load_video_timeseries(args.video, name=args.name or os.path.splitext(os.path.basename(args.video))[0])
        save_ts_to_csv(ts, args.out)
        return

    if args.text:
        ts = mmio.load_text_srt_timeseries(args.text, name=args.name or os.path.splitext(os.path.basename(args.text))[0])
        save_ts_to_csv(ts, args.out)
        return


if __name__ == '__main__':
    main()


