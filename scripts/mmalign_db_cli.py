#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import uuid
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mmalign import db as mmdb
from mmalign.data_models import TimeSeries, ModalityType


def cmd_status(args):
    try:
        conn = mmdb.get_conn_from_env()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM streams")
            n_streams = cur.fetchone()[0]
        print(f"✅ Connected. streams={n_streams}")
    except Exception as e:
        print(f"❌ DB not available: {e}")


def cmd_init(args):
    try:
        conn = mmdb.get_conn_from_env()
        with open(os.path.join(PROJECT_ROOT, 'db', 'schema.sql'), 'r', encoding='utf-8') as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("✅ Schema applied.")
    except Exception as e:
        print(f"❌ Init failed: {e}")


def cmd_import_csv(args):
    name = args.name or os.path.splitext(os.path.basename(args.csv))[0]
    df = pd.read_csv(args.csv)
    if 'time' not in df.columns:
        raise ValueError("CSV 必须包含 time 列")
    value_cols = [c for c in df.columns if c != 'time']
    if not value_cols:
        raise ValueError("CSV 必须包含至少一个数值列")
    ts = TimeSeries(
        name=name,
        modality=ModalityType.SENSOR,
        timestamps=df['time'].astype(float).to_numpy(),
        values=df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(),
    )
    conn = mmdb.get_conn_from_env()
    sid = mmdb.save_stream(conn, ts)
    print(f"✅ imported stream {name} id={sid}")


def main():
    parser = argparse.ArgumentParser(description='mmalign DB CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p0 = sub.add_parser('status', help='检查数据库连接')
    p0.set_defaults(func=cmd_status)

    p1 = sub.add_parser('init', help='初始化数据库schema')
    p1.set_defaults(func=cmd_init)

    p2 = sub.add_parser('import-csv', help='导入CSV为流（sensor）')
    p2.add_argument('--csv', required=True)
    p2.add_argument('--name', required=False)
    p2.set_defaults(func=cmd_import_csv)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()


