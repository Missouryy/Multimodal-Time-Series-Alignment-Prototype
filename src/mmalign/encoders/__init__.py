# Simple encoder stubs; extend as needed
from .simple_timeseries import extract_basic_statistics as ts_basic
from .simple_text import bag_of_words_counts as text_basic
from .simple_vision import frame_mean_rgb as vision_basic

# Advanced encoders (optional, require torch/transformers)
try:
    from .lstm_encoder import LSTMEncoder, encode_timeseries_with_lstm
    from .tcn_encoder import TCNEncoder, encode_timeseries_with_tcn
    from .bert_encoder import BERTTextEncoder, encode_text_segments_with_bert, encode_timestamped_text
    from .cnn_encoder import CNNFrameEncoder, encode_video_frames_with_cnn, simple_spatial_features
    ADVANCED_ENCODERS_AVAILABLE = True
except ImportError:
    ADVANCED_ENCODERS_AVAILABLE = False
    LSTMEncoder = None
    TCNEncoder = None
    BERTTextEncoder = None
    CNNFrameEncoder = None
    encode_timeseries_with_lstm = None
    encode_timeseries_with_tcn = None
    encode_text_segments_with_bert = None
    encode_timestamped_text = None
    encode_video_frames_with_cnn = None
    simple_spatial_features = None

__all__ = [
    "ts_basic",
    "text_basic",
    "vision_basic",
    "ADVANCED_ENCODERS_AVAILABLE",
    "LSTMEncoder",
    "TCNEncoder",
    "BERTTextEncoder",
    "CNNFrameEncoder",
    "encode_timeseries_with_lstm",
    "encode_timeseries_with_tcn",
    "encode_text_segments_with_bert",
    "encode_timestamped_text",
    "encode_video_frames_with_cnn",
    "simple_spatial_features",
]
