from __future__ import annotations

import json
import os
import uuid
from typing import Dict, Iterable, List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from .data_models import TimeSeries


def get_conn_from_env():
    """
    Read DB connection from env vars:
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    """
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5432"))
    db = os.getenv("PG_DB", "mmalign")
    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "postgres")
    return psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pwd)


def upsert_stream(conn, *, stream_id: Optional[str], name: str, modality: str, sample_rate_hz: Optional[float], metadata: Optional[Dict] = None) -> str:
    sid = stream_id or str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO streams (id, name, modality, sample_rate_hz, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
              modality = EXCLUDED.modality,
              sample_rate_hz = EXCLUDED.sample_rate_hz,
              metadata = EXCLUDED.metadata
            RETURNING id
            """,
            (sid, name, modality, sample_rate_hz, json.dumps(metadata or {})),
        )
        sid_db = cur.fetchone()[0]
    conn.commit()
    return sid_db


def insert_timeseries_values(conn, stream_id: str, timestamps: Iterable[float], values: Iterable[Iterable[float]]):
    rows = [(stream_id, float(t), list(map(float, v))) for t, v in zip(timestamps, values)]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO timeseries_values (stream_id, t, v) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()


def create_alignment(conn, *, ref_stream_id: str, tgt_stream_id: str, method: str, params: Optional[Dict] = None) -> str:
    aid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alignments (id, ref_stream_id, tgt_stream_id, method, params)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (aid, ref_stream_id, tgt_stream_id, method, json.dumps(params or {})),
        )
        aid_db = cur.fetchone()[0]
    conn.commit()
    return aid_db


def insert_alignment_overlay(conn, alignment_id: str, df: pd.DataFrame):
    if "time" not in df.columns:
        raise ValueError("overlay dataframe must contain 'time' column")
    cols = [c for c in df.columns if c != "time"]
    rows = []
    for _, row in df.iterrows():
        values = {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols}
        rows.append((alignment_id, float(row["time"]), json.dumps(values)))
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO alignment_overlays (alignment_id, t, values) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()


def save_stream(conn, ts: TimeSeries, stream_id: Optional[str] = None) -> str:
    sr = ts.rate_hz
    sid = upsert_stream(
        conn,
        stream_id=stream_id,
        name=ts.name,
        modality=ts.modality.name.lower(),
        sample_rate_hz=sr,
        metadata=ts.metadata,
    )
    insert_timeseries_values(conn, sid, ts.timestamps, ts.values if ts.values.ndim == 2 else ts.values.reshape(-1, 1))
    return sid


def get_stream_id_by_name(conn, name: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM streams WHERE name=%s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


