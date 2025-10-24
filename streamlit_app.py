import os
import sys
import tempfile
from typing import List, Optional

import pandas as pd
import streamlit as st

# Ensure local src package is importable without installation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mmalign.data_models import TimeSeries, ModalityType
from mmalign import io as mmio
from mmalign import align as mmalign
from mmalign.encoders import (
    ADVANCED_ENCODERS_AVAILABLE,
    encode_timeseries_with_lstm,
    encode_timeseries_with_tcn,
)

import datetime
from mmalign import db as mmdb
from mmalign.i18n import t


def get_lang() -> str:
    return st.session_state.get("lang", "简体中文")


def lang_selector():
    with st.sidebar:
        lang = st.selectbox("语言 / Language", ["简体中文", "English"], index=0, key="lang")
        return lang


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
    st.sidebar.header(t("sidebar.upload", lang))
    modality = st.sidebar.selectbox(
        t("sidebar.modality", lang), [m.name for m in ModalityType], index=0
    )
    allowed_types = ["csv", "wav", "mp3", "flac", "mp4", "mov", "avi", "mpg", "mpeg", "srt", "txt", "mpeg4"]
    uploaded = st.sidebar.file_uploader(
        t("sidebar.upload_help", lang),
        type=allowed_types,
    )
    stream_name = st.sidebar.text_input(t("sidebar.stream_name", lang), value="stream_1")
    if st.sidebar.button(t("sidebar.add_stream", lang), disabled=uploaded is None):
        if uploaded is None:
            st.sidebar.warning(t("msg.upload_hint", lang))
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
            mtype = ModalityType[modality]
            if mtype == ModalityType.SENSOR:
                stream = mmio.load_csv_timeseries(tmp_path, name=stream_name)
            elif mtype == ModalityType.AUDIO:
                stream = mmio.load_audio_timeseries(tmp_path, name=stream_name)
            elif mtype == ModalityType.VIDEO:
                stream = mmio.load_video_timeseries(tmp_path, name=stream_name)
            elif mtype == ModalityType.TEXT:
                stream = mmio.load_text_srt_timeseries(tmp_path, name=stream_name)
            else:
                st.sidebar.error(t("msg.unsupported_modality", lang))
                return
            add_stream(stream)
            st.sidebar.success(t("msg.added_stream", lang).format(name=stream.name, modality=modality, n=len(stream.timestamps)))
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
    if ADVANCED_ENCODERS_AVAILABLE and s.modality in (ModalityType.SENSOR, ModalityType.AUDIO):
        encoders += ["LSTM", "TCN"]
    with col2:
        enc = st.selectbox(t("encode.encoder", lang), encoders, index=0)
    with col3:
        new_name = st.text_input(t("encode.new_name", lang), value=f"{s.name}_{enc.lower()}" if enc != "None" else f"{s.name}")

    if enc == "None":
        st.caption(t("msg.no_encoder_selected", lang))
        return

    if not ADVANCED_ENCODERS_AVAILABLE:
        st.warning(t("msg.no_adv_enc", lang))
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

    if st.button(t("ui.extract_and_create", lang)):
        try:
            vals = s.values if s.values.ndim == 2 else s.values.reshape(-1, 1)
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
            new_ts = TimeSeries(
                name=new_name,
                modality=s.modality,
                timestamps=s.timestamps,
                values=feats,
                metadata={**s.metadata, "encoder": enc},
            )
            add_stream(new_ts)
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
    df = s.to_dataframe()
    st.line_chart(df.set_index("time"))
    if overlay != "None" and overlay != selected:
        s2 = get_stream(overlay)
        if s2 is not None:
            df2 = s2.to_dataframe()
            st.line_chart(df2.set_index("time"))


def alignment_ui():
    lang = get_lang()
    st.subheader(t("align.title", lang))
    names = list_stream_names()
    if len(names) < 2:
        st.info(t("msg.need_two_streams", lang))
        return
    col1, col2 = st.columns(2)
    with col1:
        ref_name = st.selectbox(t("ui.ref_stream", lang), names)
    with col2:
        tgt_name = st.selectbox(t("ui.tgt_stream", lang), [n for n in names if n != ref_name])
    ref = get_stream(ref_name)
    tgt = get_stream(tgt_name)
    method = st.selectbox(t("align.method", lang), [t("method.resample", lang), t("method.xcorr", lang), t("method.dtw", lang)])
    target_rate = st.number_input(
        t("align.rate", lang),
        min_value=1.0,
        max_value=200.0,
        value=50.0,
        step=1.0,
    )
    if method == "Cross-correlation (lag)":
        max_lag = st.number_input(
        t("align.max_lag", lang), min_value=0.1, max_value=30.0, value=5.0, step=0.1
        )
    else:
        max_lag = 5.0

    if st.button(t("align.run", lang)):
        try:
            if method == t("method.resample", lang):
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_r = mmalign.resample_to_rate(tgt, target_rate_hz=target_rate)
                overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
                st.session_state.aligned = {
                    "method": method,
                    "overlay": overlay_df,
                }
                st.success(t("msg.resampled_ok", lang))
                st.line_chart(overlay_df.set_index("time"))
            elif method == t("method.xcorr", lang):
                lag = mmalign.estimate_lag_by_xcorr(
                    ref, tgt, max_lag_seconds=max_lag
                )
                st.info(t("msg.estimated_lag", lang).format(lag=lag))
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_shift = mmalign.shift_series_in_time(tgt, shift_seconds=-lag)
                tgt_r = mmalign.resample_to_rate(tgt_shift, target_rate_hz=target_rate)
                overlay_df = mmalign.overlay_on_common_grid([ref_r, tgt_r])
                st.session_state.aligned = {
                    "method": method,
                    "lag": lag,
                    "overlay": overlay_df,
                }
                st.line_chart(overlay_df.set_index("time"))
            else:  # DTW
                ref_r = mmalign.resample_to_rate(ref, target_rate_hz=target_rate)
                tgt_r = mmalign.resample_to_rate(tgt, target_rate_hz=target_rate)
                mapping, dist = mmalign.dtw_mapping(ref_r, tgt_r)
                aligned_df = mmalign.apply_mapping_and_overlay(ref_r, tgt_r, mapping)
                st.session_state.aligned = {
                    "method": method,
                    "dtw_distance": float(dist),
                    "overlay": aligned_df,
                }
                st.success(t("msg.dtw_ok", lang).format(dist=dist))
                st.line_chart(aligned_df.set_index("time"))
        except Exception as e:
            st.error(t("msg.align_failed", lang).format(err=e))


def post_alignment_tools():
    lang = get_lang()
    st.subheader(t("post.title", lang))
    aligned = st.session_state.get("aligned", None)
    if not aligned or "overlay" not in aligned:
        st.info(t("post.no_alignment", lang))
        return
    df = aligned["overlay"]
    if df is None or len(df) == 0:
        st.warning(t("post.empty_alignment", lang))
        return
    t_min = float(df["time"].min())
    t_max = float(df["time"].max())
    sel = st.slider(t("post.time_window", lang), min_value=t_min, max_value=t_max, value=(t_min, t_max))
    view = df[(df["time"] >= sel[0]) & (df["time"] <= sel[1])]
    st.line_chart(view.set_index("time"))

    with st.expander(t("post.corr", lang), expanded=False):
        corr = df.drop(columns=["time"]).corr(numeric_only=True)
        st.dataframe(corr)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(t("post.download", lang), data=csv_bytes, file_name="aligned_overlay.csv", mime="text/csv")

    col1, col2 = st.columns(2)
    with col1:
        default_name = f"aligned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_name = st.text_input(t("post.save_outputs", lang)+" 文件名", value=default_name)
    with col2:
        if st.button(t("post.save_outputs", lang)):
            try:
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, save_name)
                df.to_csv(out_path, index=False)
                st.success(t("post.saved_to", lang).format(path=out_path))
            except Exception as e:
                st.error(t("post.save_failed", lang).format(err=e))

    st.markdown("---")
    st.subheader(t("post.save_db", lang)+"（PostgreSQL/TimescaleDB）")
    with st.expander(t("post.save_block", lang), expanded=False):
        try:
            conn = mmdb.get_conn_from_env()
            st.success(t("db.connected_detail", lang))
            # 选择要保存的参考/目标流
            names = list_stream_names()
            if len(names) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    ref_name = st.selectbox(t("db.select_ref", lang), names, key="db_ref")
                with c2:
                    tgt_name = st.selectbox(t("db.select_tgt", lang), [n for n in names if n != ref_name], key="db_tgt")
                if st.button(t("db.save_overlay", lang)):
                    try:
                        ref = get_stream(ref_name)
                        tgt = get_stream(tgt_name)
                        ref_id = mmdb.save_stream(conn, ref)
                        tgt_id = mmdb.save_stream(conn, tgt)
                        method = st.session_state.aligned.get("method", "unknown")
                        params = {k: v for k, v in st.session_state.aligned.items() if k not in {"overlay"}}
                        aid = mmdb.create_alignment(conn, ref_stream_id=ref_id, tgt_stream_id=tgt_id, method=method, params=params)
                        mmdb.insert_alignment_overlay(conn, aid, df)
                        st.success(t("db.save_ok", lang))
                    except Exception as e:
                        st.error(t("db.save_failed", lang).format(err=e))
            else:
                st.info(t("db.need_two_streams", lang))
        except Exception as e:
            st.info(t("db.not_connected_detail", lang).format(err=e))


def main():
    lang_selector()
    lang = get_lang()
    st.set_page_config(page_title=t("app.title", lang), layout="wide")
    st.title(t("app.title", lang))
    init_state()
    with st.sidebar:
        st.markdown(t("sidebar.header", lang))
        # 数据库连接状态
        try:
            conn = mmdb.get_conn_from_env()
            st.caption(t("db.connected", lang))
        except Exception:
            st.caption(t("db.not_connected", lang))
    sidebar_upload()
    visualize_streams()
    st.divider()
    feature_encoding_ui()
    st.divider()
    alignment_ui()
    st.divider()
    post_alignment_tools()


if __name__ == "__main__":
    main()
