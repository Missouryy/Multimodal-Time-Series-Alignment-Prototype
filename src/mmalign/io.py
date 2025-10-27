from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from .data_models import TimeSeries, ModalityType


# --- Helpers -----------------------------------------------------------------

def _ensure_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


# --- CSV / Sensor ------------------------------------------------------------

def load_csv_timeseries(path: str, name: str) -> TimeSeries:
    df = pd.read_csv(path)
    time_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"time", "timestamp", "t"}:
            time_col = c
            break
    if time_col is None:
        raise ValueError("CSV must contain a time or timestamp column")
    time = df[time_col].astype(float).to_numpy()
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        raise ValueError("CSV must contain at least one value column")
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isnan(values).any():
        values = np.nan_to_num(values)
    order = np.argsort(time)
    time = time[order]
    values = values[order]
    return TimeSeries(name=name, modality=ModalityType.SENSOR, timestamps=time, values=_ensure_2d(values))


# --- Audio -------------------------------------------------------------------

def load_audio_timeseries(path: str, name: str, sr: int = 16000, hop_length: int = 512) -> TimeSeries:
    import librosa

    y, sr = librosa.load(path, sr=sr, mono=True)
    # Use RMS energy as a robust envelope
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length).flatten()
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return TimeSeries(name=name, modality=ModalityType.AUDIO, timestamps=times, values=rms.reshape(-1, 1), metadata={"sr": sr, "hop_length": hop_length})


# --- Video -------------------------------------------------------------------

def load_video_timeseries(path: str, name: str, save_frames: bool = False, max_frames: int = 1000) -> TimeSeries:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Failed to open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    values: List[List[float]] = []
    times: List[float] = []
    frames: List[np.ndarray] = [] if save_frames else None
    idx = 0
    
    # 如果帧数太多，采样
    frame_skip = max(1, total_frames // max_frames) if save_frames and total_frames > max_frames else 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute mean RGB per frame
        mean_vals = frame.mean(axis=(0, 1))  # B, G, R
        # Convert BGR to RGB order
        mean_rgb = [float(mean_vals[2]), float(mean_vals[1]), float(mean_vals[0])]
        values.append(mean_rgb)
        times.append(idx / fps)
        
        # 保存原始帧（用于 CNN 编码器），BGR 转 RGB
        if save_frames and idx % frame_skip == 0 and len(frames) < max_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        idx += 1
    cap.release()
    
    if len(times) == 0:
        raise ValueError("No frames decoded from video")
    
    metadata = {"fps": fps}
    if save_frames and frames:
        metadata["frames"] = frames
        metadata["frame_indices"] = list(range(0, idx, frame_skip))[:len(frames)]
    
    return TimeSeries(
        name=name, 
        modality=ModalityType.VIDEO, 
        timestamps=np.asarray(times, dtype=float), 
        values=np.asarray(values, dtype=float), 
        metadata=metadata
    )


# --- Text (SRT or time-text) -------------------------------------------------

def _parse_srt_time(token: str) -> float:
    # Format: HH:MM:SS,mmm
    hms, ms = token.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s) + int(ms) / 1000.0


def _bin_counts(segments: List[Tuple[float, float, str]], bin_size: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if not segments:
        return np.array([]), np.array([])
    start = min(s for s, _, _ in segments)
    end = max(e for _, e, _ in segments)
    if end <= start:
        end = start + bin_size
    edges = np.arange(start, end + bin_size, bin_size)
    counts = np.zeros(len(edges) - 1, dtype=float)
    for s, e, text in segments:
        w = max(1, len(str(text).split()))
        # Distribute word count uniformly across covered bins
        left = max(s, start)
        right = min(e, end)
        if right <= left:
            # Treat as an instantaneous event
            bin_idx = int((left - start) // bin_size)
            if 0 <= bin_idx < len(counts):
                counts[bin_idx] += w
            continue
        first_bin = int((left - start) // bin_size)
        last_bin = int(np.ceil((right - start) / bin_size)) - 1
        span = max(1, last_bin - first_bin + 1)
        add = w / span
        for i in range(first_bin, last_bin + 1):
            if 0 <= i < len(counts):
                counts[i] += add
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts


def load_text_srt_timeseries(path: str, name: str, bin_size: float = 0.5) -> TimeSeries:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".srt":
        segments: List[Tuple[float, float, str]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
        i = 0
        while i < len(lines):
            # Skip index line
            if lines[i].strip().isdigit():
                i += 1
            if i >= len(lines):
                break
            if "-->" in lines[i]:
                tline = lines[i]
                i += 1
                try:
                    left, right = [p.strip() for p in tline.split("-->")]
                    s = _parse_srt_time(left)
                    e = _parse_srt_time(right)
                except Exception:
                    # Skip malformed
                    while i < len(lines) and lines[i].strip() != "":
                        i += 1
                    i += 1
                    continue
                text_lines: List[str] = []
                while i < len(lines) and lines[i].strip() != "":
                    text_lines.append(lines[i])
                    i += 1
                segments.append((s, e, " ".join(text_lines)))
            # Skip blank line
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        t, c = _bin_counts(segments, bin_size=bin_size)
        
        # 保存原始文本段落和时间戳，用于高级编码器（如 BERT）
        text_segments = [text for _, _, text in segments]
        segment_timestamps = np.array([s for s, _, _ in segments])
        
        return TimeSeries(
            name=name, 
            modality=ModalityType.TEXT, 
            timestamps=t, 
            values=c.reshape(-1, 1), 
            metadata={
                "bin_size": bin_size,
                "text_segments": text_segments,  # 原始文本内容
                "segment_timestamps": segment_timestamps,  # 文本段落的时间戳
                "segments": segments  # 完整的段落信息 (start, end, text)
            }
        )
    # Fallback: try simple time,text format (CSV or TSV) with 'time' column
    try:
        df = pd.read_csv(path)
        if "time" not in df.columns:
            # Try TSV
            df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, engine="python")
        if "time" not in df.columns:
            raise ValueError("text file must have a 'time' column")
        df = df.sort_values("time")
        
        # 保存原始文本内容（如果存在）
        text_segments = None
        if "text" in df.columns:
            text_segments = df["text"].astype(str).tolist()
        
        if "value" in df.columns:
            vals = pd.to_numeric(df["value"], errors="coerce").fillna(0).to_numpy()
        else:
            # Count words if 'text' exists
            if "text" in df.columns:
                vals = df["text"].astype(str).apply(lambda s: len(s.split())).to_numpy()
            else:
                vals = np.ones(len(df), dtype=float)
        
        metadata = {}
        if text_segments is not None:
            metadata["text_segments"] = text_segments
            metadata["segment_timestamps"] = df["time"].astype(float).to_numpy()
        
        return TimeSeries(
            name=name, 
            modality=ModalityType.TEXT, 
            timestamps=df["time"].astype(float).to_numpy(), 
            values=_ensure_2d(vals),
            metadata=metadata
        )
    except Exception as e:
        raise ValueError(f"Cannot parse text file: {e}")
