import os
import sys

# Ensure local src package is importable without installation
# MUST be before importing mmalign modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import tempfile
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from mmalign.data_models import TimeSeries, ModalityType
from mmalign import io as mmio
from mmalign import align as mmalign
from mmalign.encoders import (
    ADVANCED_ENCODERS_AVAILABLE,
    encode_timeseries_with_lstm,
    encode_timeseries_with_tcn,
    encode_video_frames_with_cnn,
    encode_text_segments_with_bert,
)

import datetime
from mmalign import db as mmdb
from mmalign.i18n import t

def get_lang() -> str:
    return st.session_state.get("lang", "简体中文")


def sidebar_settings():
    """侧边栏设置区域：语言和数据库状态"""
    lang = get_lang()
    st.sidebar.subheader(t("sidebar.settings", lang))
    
    # 语言选择
    st.sidebar.selectbox(t("sidebar.language", lang), ["简体中文", "English"], index=0, key="lang")
    
    # 数据库连接状态
    st.sidebar.markdown('##### ' + t('sidebar.db_status', lang))
    try:
        conn = mmdb.get_conn_from_env()
        st.sidebar.success(t("db.connected", lang))
    except Exception:
        st.sidebar.info(t("db.not_connected", lang))


def init_state():
    if "streams" not in st.session_state:
        st.session_state.streams = {}
    if "aligned" not in st.session_state:
        st.session_state.aligned = None


def add_stream(stream: TimeSeries):
    st.session_state.streams[stream.name] = stream


def list_stream_names() -> List[str]:
    return list(st.session_state.streams.keys())


def get_stream(name: str) -> Optional[TimeSeries]:
    return st.session_state.streams.get(name)


def sidebar_upload():
    lang = get_lang()
    st.sidebar.subheader(t("sidebar.upload", lang))
    
    allowed_types = ["csv", "wav", "mp3", "flac", "mp4", "mov", "avi", "mpg", "mpeg", "srt", "txt", "mpeg4", "tsv"]
    uploaded = st.sidebar.file_uploader(
        t("sidebar.upload_help", lang),
        type=allowed_types,
    )
    stream_name = st.sidebar.text_input(t("sidebar.stream_name", lang), value="stream_1")
    
    # 根据文件后缀自动判断模态类型和是否保存原始数据
    mtype = None
    save_raw_data = False
    
    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1].lower()
        # 音频文件
        if suffix in [".wav", ".mp3", ".flac"]:
            mtype = ModalityType.AUDIO
        # 视频文件 - 默认保存帧
        elif suffix in [".mp4", ".mov", ".avi", ".mpg", ".mpeg", ".mpeg4"]:
            mtype = ModalityType.VIDEO
            save_raw_data = True  # VIDEO默认保存帧
        # 文本文件 - 默认保存原始文本
        elif suffix in [".srt", ".txt", ".tsv"]:
            mtype = ModalityType.TEXT
            save_raw_data = True  # TEXT默认保存原始文本
        # CSV文件
        elif suffix == ".csv":
            mtype = ModalityType.SENSOR
        else:
            mtype = ModalityType.SENSOR  # 默认为传感器数据
    
    if st.sidebar.button(t("sidebar.add_stream", lang), disabled=uploaded is None):
        if uploaded is None or mtype is None:
            st.sidebar.info(t("msg.upload_hint", lang))
            return
        # 限制 200MB
        uploaded.seek(0, os.SEEK_END)
        size_mb = uploaded.tell() / (1024 * 1024)
        uploaded.seek(0)
        if size_mb > 200:
            st.sidebar.error(t("msg.file_too_large", lang).format(mb=200))
            return
        suffix = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            # 显示加载进度
            progress_placeholder = st.sidebar.empty()
            progress_bar = progress_placeholder.progress(0)
            status_placeholder = st.sidebar.empty()
            
            status_placeholder.text(("正在加载数据..." if lang == "简体中文" else "Loading data..."))
            progress_bar.progress(30)
            
            if mtype == ModalityType.SENSOR:
                stream = mmio.load_csv_timeseries(tmp_path, name=stream_name)
            elif mtype == ModalityType.AUDIO:
                stream = mmio.load_audio_timeseries(tmp_path, name=stream_name)
            elif mtype == ModalityType.VIDEO:
                stream = mmio.load_video_timeseries(tmp_path, name=stream_name, save_frames=save_raw_data)
            elif mtype == ModalityType.TEXT:
                stream = mmio.load_text_srt_timeseries(tmp_path, name=stream_name)
            else:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.sidebar.error(t("msg.unsupported_modality", lang))
                return
            
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_placeholder.empty()
            status_placeholder.empty()
            
            add_stream(stream)
            
            # 显示额外信息
            extra_info = ""
            if mtype == ModalityType.TEXT and "text_segments" in stream.metadata:
                extra_info = f" | {len(stream.metadata['text_segments'])} " + (t("msg.text_segments", lang) if "msg.text_segments" in dir(t) else ("个文本段落" if lang == "简体中文" else "text segments"))
            elif mtype == ModalityType.VIDEO and "frames" in stream.metadata:
                extra_info = f" | {len(stream.metadata['frames'])} " + (t("msg.video_frames", lang) if "msg.video_frames" in dir(t) else ("个视频帧" if lang == "简体中文" else "video frames"))
            
            st.sidebar.success(t("msg.added_stream", lang).format(name=stream.name, modality=mtype.name, n=len(stream.timestamps)) + extra_info)
        except Exception as e:
            st.sidebar.error(t("msg.failed_to_load", lang).format(err=e))
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def feature_encoding_ui():
    lang = get_lang()
    st.subheader(t("encode.title", lang))
    names = list_stream_names()
    if not names:
        st.info(t("msg.need_one_stream_encode", lang))
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        sel = st.selectbox(t("msg.select_stream_to_encode", lang), names)
    s = get_stream(sel)
    if s is None:
        return
    encoders = ["None"]
    if ADVANCED_ENCODERS_AVAILABLE:
        if s.modality in (ModalityType.SENSOR, ModalityType.AUDIO):
            encoders += ["LSTM", "TCN"]
        elif s.modality == ModalityType.VIDEO:
            encoders += ["CNN (ResNet)"]
        elif s.modality == ModalityType.TEXT:
            encoders += ["BERT"]
    with col2:
        enc = st.selectbox(t("encode.encoder", lang), encoders, index=0)
    with col3:
        new_name = st.text_input(t("encode.new_name", lang), value=f"{s.name}_{enc.lower().replace(' ', '_').replace('(', '').replace(')', '')}" if enc != "None" else f"{s.name}")

    if enc == "None":
        st.caption(t("msg.no_encoder_selected", lang))
        return

    if not ADVANCED_ENCODERS_AVAILABLE:
        st.error(t("msg.no_adv_enc", lang))
        return

    # 参数区
    with st.expander(t("ui.advanced_params", lang), expanded=False):
        if enc == "LSTM":
            lstm_hidden = st.number_input("hidden_dim", min_value=8, max_value=256, value=64, step=8)
            lstm_layers = st.number_input("num_layers", min_value=1, max_value=4, value=2, step=1)
            lstm_bi = st.checkbox("bidirectional", value=True)
        elif enc == "TCN":
            tcn_channels_text = st.text_input(t("ui.num_channels_list", lang), value="64,64,128")
            tcn_kernel = st.number_input("kernel_size", min_value=2, max_value=9, value=3, step=1)
            tcn_dropout = st.slider("dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
        elif enc == "CNN (ResNet)":
            cnn_model = st.selectbox("model_name", ["resnet18", "resnet50"], index=0)
        elif enc == "BERT":
            bert_model = st.selectbox("model_name", ["bert-base-uncased", "bert-base-chinese", "distilbert-base-uncased"], index=0)

    if st.button(t("ui.extract_and_create", lang)):
        try:
            # 创建进度条容器
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 初始化变量
            feats = None
            new_timestamps = s.timestamps  # 默认使用原始时间戳
            
            if enc in ["LSTM", "TCN"]:
                vals = s.values if s.values.ndim == 2 else s.values.reshape(-1, 1)
                encoder_name = "LSTM" if enc == "LSTM" else "TCN"
                
                # 步骤1: 准备数据
                status_text.text((f"准备数据 ({len(vals)} 个时间步)..." if lang == "简体中文" else f"Preparing data ({len(vals)} timesteps)..."))
                progress_bar.progress(10)
                
                # 步骤2: 特征提取
                status_text.text((f"使用 {encoder_name} 提取特征..." if lang == "简体中文" else f"Extracting features with {encoder_name}..."))
                progress_bar.progress(30)
                
                if enc == "LSTM":
                    feats = encode_timeseries_with_lstm(
                        vals,
                        timestamps=s.timestamps,
                        hidden_dim=int(lstm_hidden),
                        num_layers=int(lstm_layers),
                        bidirectional=bool(lstm_bi),
                    )
                else:
                    channels = [int(x.strip()) for x in tcn_channels_text.split(",") if x.strip()]
                    feats = encode_timeseries_with_tcn(
                        vals,
                        timestamps=s.timestamps,
                        num_channels=channels,
                        kernel_size=int(tcn_kernel),
                        dropout=float(tcn_dropout),
                    )
                
                progress_bar.progress(70)
                
            elif enc == "CNN (ResNet)":
                # 检查是否有原始帧数据
                frames = s.metadata.get("frames", None)
                frame_indices = s.metadata.get("frame_indices", None)
                if frames is None or len(frames) == 0:
                    status_text.empty()
                    progress_bar.empty()
                    st.info(("CNN 编码器需要原始视频帧数据。请在元数据中提供 'frames' 字段。" if lang == "简体中文" else "CNN encoder requires original video frames. Please provide 'frames' in metadata."))
                    return
                
                # 步骤1: 准备数据
                status_text.text((f"准备 {len(frames)} 帧视频数据..." if lang == "简体中文" else f"Preparing {len(frames)} video frames..."))
                progress_bar.progress(10)
                
                # 步骤2: 特征提取
                status_text.text((f"使用 {cnn_model} 提取特征..." if lang == "简体中文" else f"Extracting features with {cnn_model}..."))
                progress_bar.progress(30)
                
                feats = encode_video_frames_with_cnn(frames, model_name=cnn_model)
                
                # 使用对应帧的时间戳
                if frame_indices is not None and len(frame_indices) == len(feats):
                    new_timestamps = s.timestamps[frame_indices]
                else:
                    # 如果没有 frame_indices，均匀采样时间戳
                    indices = np.linspace(0, len(s.timestamps) - 1, len(feats), dtype=int)
                    new_timestamps = s.timestamps[indices]
                
                progress_bar.progress(70)
                
            elif enc == "BERT":
                # 检查是否有原始文本数据
                text_segments = s.metadata.get("text_segments", None)
                segment_timestamps = s.metadata.get("segment_timestamps", None)
                if text_segments is None or len(text_segments) == 0:
                    status_text.empty()
                    progress_bar.empty()
                    st.info(("BERT 编码器需要原始文本内容。请在元数据中提供 'text_segments' 字段。" if lang == "简体中文" else "BERT encoder requires original text content. Please provide 'text_segments' in metadata."))
                    return
                
                # 步骤1: 准备数据
                status_text.text((f"准备 {len(text_segments)} 个文本片段..." if lang == "简体中文" else f"Preparing {len(text_segments)} text segments..."))
                progress_bar.progress(10)
                
                # 步骤2: 特征提取
                status_text.text((f"使用 {bert_model} 提取特征..." if lang == "简体中文" else f"Extracting features with {bert_model}..."))
                progress_bar.progress(30)
                
                feats = encode_text_segments_with_bert(text_segments, model_name=bert_model)
                
                # 使用文本段落的时间戳
                if segment_timestamps is not None and len(segment_timestamps) == len(feats):
                    new_timestamps = segment_timestamps
                else:
                    # 如果没有 segment_timestamps，均匀采样时间戳
                    indices = np.linspace(0, len(s.timestamps) - 1, len(feats), dtype=int)
                    new_timestamps = s.timestamps[indices]
                
                progress_bar.progress(70)
                
            else:
                status_text.text(("未知的编码器类型" if lang == "简体中文" else "Unknown encoder type"))
                progress_bar.empty()
                return
            
            # 步骤3: 创建新数据流
            status_text.text(("创建新数据流..." if lang == "简体中文" else "Creating new stream..."))
            progress_bar.progress(85)
            
            new_ts = TimeSeries(
                name=new_name,
                modality=s.modality,
                timestamps=new_timestamps,
                values=feats,
                metadata={**s.metadata, "encoder": enc},
            )
            add_stream(new_ts)
            
            # 步骤4: 完成
            progress_bar.progress(100)
            status_text.text(("完成！" if lang == "简体中文" else "Done!"))
            
            # 清理进度显示
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success(t("msg.encoding_success", lang).format(name=new_ts.name, shape=new_ts.values.shape))
        except Exception as e:
            st.error(t("msg.encoding_failed", lang).format(err=e))


def visualize_streams():
    lang = get_lang()
    st.subheader(t("streams.title", lang))
    names = list_stream_names()
    if not names:
        st.info(t("msg.upload_streams_to_begin", lang))
        return
    col1, col2 = st.columns(2)
    with col1:
        selected = st.selectbox(t("streams.select", lang), names)
    with col2:
        overlay = st.selectbox(t("streams.overlay_with", lang), ["None"] + names, index=0)
    s = get_stream(selected)
    if s is None:
        return
    
    # 显示流的基本信息和元数据
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("模态" if lang == "简体中文" else "Modality", s.modality.name)
    with info_cols[1]:
        st.metric("数据点" if lang == "简体中文" else "Data points", len(s.timestamps))
    with info_cols[2]:
        st.metric("维度" if lang == "简体中文" else "Dimensions", s.values.shape[1] if s.values.ndim > 1 else 1)
    with info_cols[3]:
        st.metric("时长 (秒)" if lang == "简体中文" else "Duration (s)", f"{s.duration:.2f}")
    
    # 显示高级编码器支持信息
    if s.modality == ModalityType.TEXT and "text_segments" in s.metadata:
        st.success(("包含 {n} 个文本段落，可使用 BERT 编码器" if lang == "简体中文" else "Contains {n} text segments, BERT encoder available").format(n=len(s.metadata['text_segments'])))
    elif s.modality == ModalityType.VIDEO and "frames" in s.metadata:
        st.success(("包含 {n} 个视频帧，可使用 CNN 编码器" if lang == "简体中文" else "Contains {n} video frames, CNN encoder available").format(n=len(s.metadata['frames'])))
    elif s.modality == ModalityType.TEXT:
        st.info(("TEXT 数据已加载，未保存原始文本段落" if lang == "简体中文" else "TEXT data loaded without raw text segments"))
    elif s.modality == ModalityType.VIDEO:
        st.info(("VIDEO 数据已加载，未保存原始视频帧" if lang == "简体中文" else "VIDEO data loaded without raw video frames"))
    
    df = s.to_dataframe()
    # 处理可能的列名问题
    if "time" in df.columns:
        st.line_chart(df.set_index("time"))
    else:
        st.line_chart(df)
    
    if overlay != "None" and overlay != selected:
        s2 = get_stream(overlay)
        if s2 is not None:
            df2 = s2.to_dataframe()
            if "time" in df2.columns:
                st.line_chart(df2.set_index("time"))
            else:
                st.line_chart(df2)


def alignment_and_analysis_ui():
    """合并的对齐与结果分析界面"""
    lang = get_lang()
    st.subheader(t("align.title", lang))
    names = list_stream_names()
    if len(names) < 2:
        st.info(t("msg.need_two_streams", lang))
        return
    
    # 对齐参数设置
    col1, col2 = st.columns(2)
    with col1:
        ref_name = st.selectbox(t("ui.ref_stream", lang), names)
    with col2:
        tgt_name = st.selectbox(t("ui.tgt_stream", lang), [n for n in names if n != ref_name])
    ref = get_stream(ref_name)
    tgt = get_stream(tgt_name)
    method = st.selectbox(t("align.method", lang), [t("method.resample", lang), t("method.xcorr", lang), t("method.dtw", lang)])
    
    col_rate, col_lag = st.columns(2)
    with col_rate:
        target_rate = st.number_input(
            t("align.rate", lang),
            min_value=1.0,
            max_value=200.0,
            value=50.0,
            step=1.0,
        )
    with col_lag:
        if method == t("method.xcorr", lang):
            max_lag = st.number_input(
                t("align.max_lag", lang), min_value=0.1, max_value=30.0, value=5.0, step=0.1
            )
        else:
            max_lag = 5.0

    if st.button(t("align.run", lang)):
        try:
            if method == t("method.resample", lang):
                # 重采样进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(("正在重采样..." if lang == "简体中文" else "Resampling..."))
                progress_bar.progress(50)
                
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_r = mmalign.resample_to_rate(tgt, target_rate_hz=target_rate)
                overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.aligned = {
                    "method": method,
                    "ref_name": ref_name,
                    "tgt_name": tgt_name,
                    "overlay": overlay_df,
                }
                st.success(t("msg.resampled_ok", lang))
            elif method == t("method.xcorr", lang):
                # 互相关对齐进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(("计算互相关..." if lang == "简体中文" else "Computing cross-correlation..."))
                progress_bar.progress(20)
                
                lag = mmalign.estimate_lag_by_xcorr(
                    ref, tgt, max_lag_seconds=max_lag
                )
                
                progress_bar.progress(50)
                status_text.text(("对齐数据..." if lang == "简体中文" else "Aligning data..."))
                
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_shift = mmalign.shift_series_in_time(tgt, shift_seconds=-lag)
                tgt_r = mmalign.resample_to_rate(tgt_shift, target_rate_hz=target_rate)
                overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
                
                progress_bar.progress(100)
                status_text.text(("完成！" if lang == "简体中文" else "Done!"))
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.info(t("msg.estimated_lag", lang).format(lag=lag))
                
                st.session_state.aligned = {
                    "method": method,
                    "lag": lag,
                    "ref_name": ref_name,
                    "tgt_name": tgt_name,
                    "overlay": overlay_df,
                }
            else:  # DTW
                # DTW 对齐进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(("重采样数据..." if lang == "简体中文" else "Resampling data..."))
                progress_bar.progress(15)
                
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_r = mmalign.resample_to_rate(tgt, target_rate_hz=target_rate)
                
                progress_bar.progress(30)
                status_text.text(("执行 DTW 对齐..." if lang == "简体中文" else "Computing DTW alignment..."))
                
                mapping, dist = mmalign.dtw_mapping(ref_r, tgt_r)
                
                progress_bar.progress(75)
                status_text.text(("生成对齐结果..." if lang == "简体中文" else "Generating alignment results..."))
                
                aligned_df = mmalign.apply_mapping_and_overlay(ref_r, tgt_r, mapping)
                
                progress_bar.progress(100)
                status_text.text(("完成！" if lang == "简体中文" else "Done!"))
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.aligned = {
                    "method": method,
                    "dtw_distance": float(dist),
                    "ref_name": ref_name,
                    "tgt_name": tgt_name,
                    "overlay": aligned_df,
                }
                st.success(t("msg.dtw_ok", lang).format(dist=dist))
        except Exception as e:
            st.error(t("msg.align_failed", lang).format(err=e))
            return

    # 结果分析与导出（自动显示）
    aligned = st.session_state.get("aligned", None)
    if aligned and "overlay" in aligned:
        st.markdown("---")
        st.subheader(t("post.title", lang))
        
        df = aligned["overlay"]
        if df is None or len(df) == 0:
            st.info(t("post.empty_alignment", lang))
            return
        
        # 对齐信息
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric(("方法" if lang == "简体中文" else "Method"), aligned.get("method", "N/A"))
        with info_cols[1]:
            if "lag" in aligned:
                st.metric(("时延(秒)" if lang == "简体中文" else "Lag (s)"), f"{aligned['lag']:.3f}")
            elif "dtw_distance" in aligned:
                st.metric(("DTW距离" if lang == "简体中文" else "DTW distance"), f"{aligned['dtw_distance']:.2f}")
            else:
                st.metric(("时延" if lang == "简体中文" else "Lag"), "N/A")
        with info_cols[2]:
            st.metric(("数据点数" if lang == "简体中文" else "Data points"), len(df))
        
        # 可视化
        st.write("")  # 添加空行
        
        # 创建标签页
        tab1, tab2 = st.tabs([
            ("时序图" if lang == "简体中文" else "Time Series"),
            (t("post.corr", lang))
        ])
        
        with tab1:
            # 数据流信息
            stream_cols = [col for col in df.columns if col != "time"]
            num_streams = len(stream_cols)
            
            # 获取数据流类型以确定单位
            ref_stream = get_stream(aligned.get("ref_name"))
            tgt_stream = get_stream(aligned.get("tgt_name"))
            
            # 根据数据模态确定纵轴单位
            y_unit = ""
            if ref_stream:
                if ref_stream.modality == ModalityType.AUDIO:
                    y_unit = "RMS能量" if lang == "简体中文" else "RMS Energy"
                elif ref_stream.modality == ModalityType.VIDEO:
                    y_unit = "RGB值(0-255)" if lang == "简体中文" else "RGB Value (0-255)"
                elif ref_stream.modality == ModalityType.TEXT:
                    y_unit = "词频" if lang == "简体中文" else "Word Count"
                elif ref_stream.modality == ModalityType.SENSOR:
                    y_unit = "传感器读数" if lang == "简体中文" else "Sensor Reading"
            
            if not y_unit:
                y_unit = "数值" if lang == "简体中文" else "Value"
            
            info_text = (
                f"共 {num_streams} 个数据通道 | 横轴：时间（秒） | 纵轴：{y_unit}" 
                if lang == "简体中文" 
                else f"{num_streams} channels | X: Time (s) | Y: {y_unit}"
            )
            st.caption(info_text)
            
            # 时间范围选择
            t_min = float(df["time"].min())
            t_max = float(df["time"].max())
            
            sel = st.slider(
                t("post.time_window", lang), 
                min_value=t_min, 
                max_value=t_max, 
                value=(t_min, t_max),
                label_visibility="collapsed"
            )
            
            # 图表
            view = df[(df["time"] >= sel[0]) & (df["time"] <= sel[1])]
            if len(view) > 0:
                st.line_chart(view.set_index("time"), use_container_width=True)
            else:
                st.info(("所选时间范围内无数据" if lang == "简体中文" else "No data in selected time range"))
        
        with tab2:
            st.caption(("数据通道间的相关性系数" if lang == "简体中文" else "Correlation coefficients between channels"))
            corr = df.drop(columns=["time"]).corr(numeric_only=True)
            st.dataframe(corr, use_container_width=True)

        # 导出选项
        st.write("")  # 添加空行
        
        # 文件名输入
        default_name = f"aligned_{aligned.get('ref_name', 'ref')}_{aligned.get('tgt_name', 'tgt')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_filename = st.text_input(
            ("导出文件" if lang == "简体中文" else "Export file"),
            value=default_name,
            help=("重命名文件" if lang == "简体中文" else "Rename the file")
        )
        
        # 三列导出按钮
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                t("post.download", lang), 
                data=csv_bytes, 
                file_name=save_filename, 
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            if st.button(("保存到 outputs/" if lang == "简体中文" else "Save to outputs/"), 
                        use_container_width=True):
                try:
                    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, save_filename)
                    df.to_csv(out_path, index=False)
                    st.success(t("post.saved_to", lang).format(path=out_path))
                except Exception as e:
                    st.error(t("post.save_failed", lang).format(err=e))
        
        with col_export3:
            try:
                conn = mmdb.get_conn_from_env()
                if st.button(("保存到数据库" if lang == "简体中文" else "Save to database"), 
                            use_container_width=True):
                    try:
                        ref = get_stream(aligned.get("ref_name"))
                        tgt = get_stream(aligned.get("tgt_name"))
                        if ref and tgt:
                            ref_id = mmdb.save_stream(conn, ref)
                            tgt_id = mmdb.save_stream(conn, tgt)
                            method = aligned.get("method", "unknown")
                            params = {k: v for k, v in aligned.items() if k not in {"overlay", "ref_name", "tgt_name"}}
                            aid = mmdb.create_alignment(conn, ref_stream_id=ref_id, tgt_stream_id=tgt_id, method=method, params=params)
                            mmdb.insert_alignment_overlay(conn, aid, df)
                            st.success(t("db.save_ok", lang))
                        else:
                            st.error(("无法获取参考流或目标流" if lang == "简体中文" else "Cannot retrieve reference or target stream"))
                    except Exception as e:
                        st.error(t("db.save_failed", lang).format(err=e))
            except Exception:
                # 数据库未连接时禁用按钮
                st.button(("保存到数据库" if lang == "简体中文" else "Save to database"), 
                         disabled=True,
                         use_container_width=True,
                         help=("数据库未连接" if lang == "简体中文" else "Database not connected"))




def main():
    lang = get_lang()
    st.set_page_config(page_title=t("app.title", lang), layout="wide")
    st.title(t("app.title", lang))
    init_state()
    
    # 侧边栏
    sidebar_upload()
    sidebar_settings()
    
    # 主界面
    visualize_streams()
    st.divider()
    feature_encoding_ui()
    st.divider()
    alignment_and_analysis_ui()


if __name__ == "__main__":
    main()
