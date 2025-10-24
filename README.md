# 多模态时序数据对齐原型系统

基于 Streamlit 的多模态时序数据对齐原型系统，支持音频、视频、传感器 CSV、带时间戳的文本数据的接入、管理、可视化和对齐。提供重采样、互相关时延估计和动态时间规整(DTW)等对齐方法，为行为识别、事件检测等上层应用提供高质量的基础数据。

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [高级编码器](#高级编码器)
- [技术架构](#技术架构)
- [应用场景](#应用场景)
- [常见问题](#常见问题)

## 功能特性

### 核心功能
- 📥 **多模态数据接入**：支持传感器、音频、视频、文本等多种模态
- 📊 **实时可视化**：波形展示、特征图、多流叠加对比
- 🔄 **三种对齐方法**：
  - **均匀重采样**：统一采样率
  - **互相关对齐**：自动时延估计
  - **动态时间规整(DTW)**：非线性时序对齐
- 🧠 **特征提取器**：
  - 音频：RMS 能量、MFCC 统计
  - 视频：帧级 RGB 特征
  - 传感器：多变量时间序列
  - 文本：时间窗口词频

### 高级编码器（可选）
- **LSTM**：双向 LSTM 捕捉时序依赖
- **TCN**：扩张卷积网络，长时依赖建模
- **BERT**：预训练语义嵌入
- **CNN (ResNet)**：视频帧特征提取

## 快速开始

### 环境要求
- macOS 10.15+ (Catalina 及以上)
- Python 3.10+
- 8GB+ 内存（推荐 16GB）

### 安装步骤

**方式一：Conda 环境（推荐）**

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate mmalign

# 3. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__)"
```

**方式二：pip + venv**

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装基础依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. 安装高级编码器（可选）
pip install torch torchvision transformers
```

> 💡 **详细安装说明**：查看 [INSTALL.md](INSTALL.md) 获取完整指南和故障排除

### 启动应用

```bash
# 启动 Streamlit Web 界面
streamlit run streamlit_app.py
```

### 数据库（可选）快速操作

如需持久化到 PostgreSQL/TimescaleDB，请先在环境中设置 PG_* 变量并执行建表（详见 INSTALL.md）。常用 CLI：

```bash
# 查看数据库状态
python scripts/mmalign_db_cli.py status

# 初始化/更新 schema（可重复执行）
python scripts/mmalign_db_cli.py init

# 将 CSV 导入为传感器流（CSV 需包含 time 列）
python scripts/mmalign_db_cli.py import-csv --csv path/to/sensor.csv --name sensor1
```

### 数据预处理与命令行对齐

```bash
# 将多模态文件预处理为标准 CSV(time,v*)
python scripts/preprocess_multimodal.py --audio path/to/audio.wav --out outputs/audio.csv
python scripts/preprocess_multimodal.py --video path/to/video.mp4 --out outputs/video.csv
python scripts/preprocess_multimodal.py --text  path/to/subs.srt  --out outputs/text.csv
python scripts/preprocess_multimodal.py --sensor path/to/sensor.csv --out outputs/sensor.csv

# 使用命令行进行对齐
python scripts/align_cli.py --ref outputs/audio.csv --tgt outputs/video.csv --method xcorr --rate 50 --out outputs/av_overlay.csv
```

## 使用说明

### 1. 上传数据

**支持格式**：

| 模态 | 文件格式 | 要求 |
|------|---------|------|
| **传感器** | CSV | 必须包含 `time` 或 `timestamp` 列 |
| **音频** | WAV, MP3, FLAC | 自动提取 RMS 特征 |
| **视频** | MP4, MOV, AVI | 自动提取帧特征 |
| **文本** | SRT, CSV (含 time 列) | 字幕或时间戳文本 |

**操作步骤**：
1. 左侧边栏选择模态类型
2. 点击上传文件
3. 输入数据流名称
4. 点击"Add stream"

### 2. 可视化

- 下拉选择要查看的数据流
- 可选择第二个流进行叠加对比
- 支持缩放、平移等交互操作

### 3. 对齐

**步骤**：
1. 选择参考流（Reference）和目标流（Target）
2. 选择对齐方法：
   - **Resample**：简单重采样（适合采样率不同但无时延）
   - **Cross-correlation**：互相关（适合固定时延）
   - **DTW**：动态时间规整（适合非线性时间变化）
3. 设置参数（目标频率、最大时延等）
4. 点击"Run alignment"执行
5. 查看对齐结果和可视化

## 高级编码器

系统提供深度学习编码器用于提取高级特征：

### LSTM 时序编码器
```python
from mmalign.encoders import encode_timeseries_with_lstm

# 编码传感器数据
features = encode_timeseries_with_lstm(
    sensor_data,
    hidden_dim=64,
    num_layers=2,
    bidirectional=True
)
# 输出: (N, 128) 特征矩阵
```

**适用场景**：
- 规则或不规则采样的时序数据
- 需要隐状态解释的场景
- IMU、生理信号等传感器数据

### TCN 时序编码器
```python
from mmalign.encoders import encode_timeseries_with_tcn

# 编码长时序数据
features = encode_timeseries_with_tcn(
    sensor_data,
    num_channels=[64, 64, 128],
    kernel_size=3
)
# 输出: (N, 128) 特征矩阵
```

**适用场景**：
- 长序列数据（>1000 个时间步）
- 需要并行训练加速
- 实时应用（低延迟）

### BERT 文本编码器
```python
from mmalign.encoders import encode_text_segments_with_bert

# 编码字幕片段
embeddings = encode_text_segments_with_bert(
    texts=["第一句", "第二句"],
    model_name="bert-base-chinese"
)
# 输出: (N, 768) 语义嵌入
```

**适用场景**：
- 字幕、转录文本
- 需要上下文语义
- 跨模态检索（文本-视频）

### CNN 视觉编码器
```python
from mmalign.encoders import encode_video_frames_with_cnn

# 编码视频帧
features = encode_video_frames_with_cnn(
    frames,  # list of RGB arrays
    model_name="resnet18"
)
# 输出: (N, 512) 空间特征
```

**适用场景**：
- 视频场景理解
- 动作识别
- 视频-音频对齐



## 技术架构

### 数据模型
```python
class TimeSeries:
    name: str                    # 数据流名称
    modality: ModalityType       # SENSOR/AUDIO/VIDEO/TEXT
    timestamps: np.ndarray       # (N,) 时间戳
    values: np.ndarray          # (N, D) 特征矩阵
    metadata: Dict              # 元数据
```

### 核心模块
- **`data_models.py`**: TimeSeries 和 ModalityType 定义
- **`io.py`**: 多模态数据加载器（CSV/音频/视频/文本）
- **`align.py`**: 对齐算法（重采样/互相关/DTW）
- **`encoders/`**: 特征提取器（简单 + 深度学习）

### 对齐方法对比

| 方法 | 时间复杂度 | 适用场景 | 优点 | 局限 |
|------|-----------|---------|------|------|
| **重采样** | O(N) | 采样率不同，无时延 | 快速简单 | 无法处理时延 |
| **互相关** | O(N log N) | 固定全局时延 | 自动检测时延 | 假设线性时移 |
| **DTW** | O(N²) | 非线性时间变化 | 处理复杂扭曲 | 计算开销大 |

## 应用场景

### 1. 行为识别
**问题**：加速度计(100Hz)、陀螺仪(100Hz)、视频(30fps) 时间不同步

**解决方案**：
1. 上传三个数据流
2. 互相关估计时延
3. 重采样到 50Hz
4. LSTM 编码对齐特征
5. 分类器识别行为

### 2. 事件检测
**问题**：音频、视频、字幕需同步以检测事件

**解决方案**：
1. 加载音频(RMS)、视频(帧特征)、字幕(BERT)
2. DTW 处理非线性变化
3. 多模态峰值检测

### 3. 传感器融合
**问题**：GPS(1Hz) 和 IMU(100Hz) 采样率差异大

**解决方案**：
1. 重采样到 10Hz
2. 对齐到公共时间戳
3. 拼接特征用于定位

## 常见问题

### 安装相关

**Q: 如何在 Mac M1/M2 上安装？**  
A: 使用 Conda 环境，PyTorch 会自动选择 Apple Silicon 优化版本：
```bash
conda env create -f environment.yml
```

**Q: FFmpeg 安装失败？**  
A (Conda): `conda install -c conda-forge ffmpeg`  
A (Homebrew): `brew install ffmpeg`

**Q: 提示"高级编码器不可用"？**  
A: 安装 PyTorch: `pip install torch torchvision transformers`

### 使用相关

**Q: 音频只显示一维特征？**  
A: 默认提取 RMS 能量。修改 `io.py` 中 `load_audio_timeseries()` 可提取更多特征（MFCC 等）

**Q: DTW 太慢？**  
A: 先降采样到 10-20Hz，或使用 FastDTW 的 radius 参数

**Q: 如何同时对齐 3 个以上数据流？**  
A: 代码中调用 `align.overlay_on_common_grid([s1, s2, s3, ...])`

**Q: 如何导出对齐结果？**  
A: 在 Streamlit 中添加：
```python
csv = overlay_df.to_csv(index=False)
st.download_button("下载", csv, "aligned.csv")
```

**Q: 视频文件太大？**  
A: 修改 `io.py` 添加跳帧，或只处理关键片段

### 性能优化

**Q: 内存不足？**  
A: 
- 降低目标采样率
- 视频分段处理
- 降维（PCA）

**Q: GPU 未被使用？**  
A: 验证 CUDA：`python -c "import torch; print(torch.cuda.is_available())"`

## 项目结构

```
.
├── README.md                     # 本文档
├── INSTALL.md                    # 安装与使用指南
├── environment.yml               # Conda 环境
├── requirements.txt              # pip 依赖
├── streamlit_app.py             # Web 应用入口
├── db/
│   └── schema.sql               # PostgreSQL/TimescaleDB 建表脚本
├── src/mmalign/                 # 核心包
│   ├── data_models.py           # 数据模型
│   ├── io.py                    # 数据加载
│   ├── align.py                 # 对齐算法
│   ├── db.py                    # 数据库读写
│   └── encoders/                # 编码器
│       ├── lstm_encoder.py      # LSTM
│       ├── tcn_encoder.py       # TCN
│       ├── bert_encoder.py      # BERT
│       ├── cnn_encoder.py       # CNN
│       └── simple_*.py          # 简单特征
└── scripts/
    └── mmalign_db_cli.py        # 数据库 CLI 工具
```

## 扩展开发

### 添加新编码器

1. 创建 `src/mmalign/encoders/my_encoder.py`
2. 实现编码函数：
```python
def my_encoder(values: np.ndarray) -> np.ndarray:
    # 编码逻辑
    return encoded_features
```
3. 在 `encoders/__init__.py` 中导出
4. 更新文档

### 添加新对齐方法

在 `src/mmalign/align.py` 中添加：
```python
def my_align(ref: TimeSeries, tgt: TimeSeries) -> TimeSeries:
    # 对齐逻辑
    return aligned_tgt
```

## 学术参考

本系统实现基于以下研究：
- **DTW 对齐**: Müller (2007) - Information Retrieval for Music and Motion
- **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
- **TCN**: Bai et al. (2018) - Temporal Convolutional Networks
- **BERT**: Devlin et al. (2019) - Pre-training of Deep Bidirectional Transformers
- **ResNet**: He et al. (2016) - Deep Residual Learning
- **多模态融合**: Zadeh et al. (2017) - Multi-attention Fusion

相关论文包含在项目目录中。

## 许可与联系

- **许可证**：用于研究和原型开发
- **联系方式**：通过 GitHub Issues 反馈问题

---

**提示**：首次使用建议先运行示例数据熟悉流程，再上传实际数据。

**系统要求**：
- macOS 10.15+（推荐 13.0+）
- 磁盘空间：~5GB（含 PyTorch 和模型）
- 内存：8GB 最低，16GB 推荐
- GPU：可选（M1/M2 Metal 或 NVIDIA CUDA）
