# 安装指南

多模态时序数据对齐系统的完整安装说明（针对 macOS 优化）。

## 系统要求

- **操作系统**: macOS 10.15+ (Catalina 或更高版本)
- **处理器**: Intel 或 Apple Silicon (M1/M2/M3)
- **内存**: 8GB 最低，16GB 推荐
- **磁盘空间**: ~5GB（含 PyTorch 和预训练模型）
- **Python**: 3.10 或更高版本

## 前置准备

### 1. 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装 Python 和 Conda（推荐 Miniforge）

**选项 A：Miniforge（推荐，支持 Apple Silicon）**

```bash
# 下载 Miniforge
brew install miniforge

# 初始化
conda init zsh  # 或 bash
```

**选项 B：使用系统 Python**

```bash
# macOS 自带 Python 3，或通过 Homebrew 安装
brew install python@3.10
```

## 安装方式

### 方式一：Conda 环境（强烈推荐）

Conda 提供完整的依赖管理，包括 FFmpeg 和 PyTorch。

```bash
# 1. 进入项目目录
cd "/Users/yanyu/Desktop/Multimodal Data Alignment"

# 2. 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate mmalign

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

### 方式二：pip + venv

适合不使用 Conda 的场景。

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活环境
source .venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装基础依赖
pip install -r requirements.txt

# 5. 安装 FFmpeg（必需，用于音视频处理）
brew install ffmpeg

# 6. 安装高级编码器（可选）
# Apple Silicon (M1/M2)
pip install torch torchvision torchaudio

# Intel Mac
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 Transformers
pip install transformers sentencepiece tokenizers
```

## 验证安装

运行以下脚本验证所有组件正常：

```bash
# 激活环境
conda activate mmalign  # 或 source .venv/bin/activate

# 运行验证
python << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"位置: {sys.executable}\n")

# 核心库
import numpy as np
import pandas as pd
import scipy
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ SciPy: {scipy.__version__}")

# 音视频处理
import librosa
import cv2
print(f"✅ Librosa: {librosa.__version__}")
print(f"✅ OpenCV: {cv2.__version__}")

# Streamlit
import streamlit
print(f"✅ Streamlit: {streamlit.__version__}")

# 深度学习（可选）
try:
    import torch
    print(f"\n✅ PyTorch: {torch.__version__}")
    print(f"   设备: {torch.backends.mps.is_available() and 'Apple Silicon GPU (MPS)' or 'CPU'}")
    if torch.backends.mps.is_available():
        print(f"   MPS 加速: 可用")
except ImportError:
    print("\n⚠️  PyTorch 未安装（高级编码器不可用）")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("⚠️  Transformers 未安装（BERT 不可用）")

print("\n✅ 安装验证完成！")
EOF
```

## 启动应用

### Streamlit Web 界面

```bash
# 确保环境已激活
conda activate mmalign

# 启动应用
streamlit run streamlit_app.py

# 应用将在浏览器中打开: http://localhost:8501
```

## 数据库（可选：持久化对齐与特征）

若需持久化流、特征与对齐结果，推荐使用 PostgreSQL + TimescaleDB。

1) 安装 PostgreSQL（可选 TimescaleDB 扩展）并创建数据库：
```bash
createdb mmalign
```

2) 执行建表脚本：
```bash
psql -U postgres -d mmalign -f db/schema.sql
```

3) 配置环境变量（Streamlit 与脚本会自动读取）：
```bash
export PG_HOST=127.0.0.1
export PG_PORT=5432
export PG_DB=mmalign
export PG_USER=postgres
export PG_PASSWORD=postgres
```

4) Python 使用示例：
```python
from mmalign.db import get_conn_from_env, save_stream, create_alignment, insert_alignment_overlay

conn = get_conn_from_env()
sid_ref = save_stream(conn, ref_timeseries)
sid_tgt = save_stream(conn, tgt_timeseries)
aid = create_alignment(conn, ref_stream_id=sid_ref, tgt_stream_id=sid_tgt, method="dtw", params={"dtw_distance": 1.23})
insert_alignment_overlay(conn, aid, overlay_df)  # overlay_df 需包含 'time' 列
```

5) 命令行工具（CLI）：

```bash
# 查看数据库状态
python scripts/mmalign_db_cli.py status

# 初始化 schema（重复执行安全）
python scripts/mmalign_db_cli.py init

# 将 CSV 导入为传感器流（需包含 time 列）
python scripts/mmalign_db_cli.py import-csv --csv path/to/sensor.csv --name sensor1
```

## 常见问题

### 1. Apple Silicon (M1/M2) 相关

**问题**: PyTorch 未使用 GPU 加速

**解决方案**:
```bash
# 验证 MPS (Metal Performance Shaders) 可用
python -c "import torch; print(torch.backends.mps.is_available())"

# 如返回 False，重新安装 PyTorch
conda install pytorch torchvision torchaudio -c pytorch
```

**问题**: Conda 安装速度慢

**解决方案**:
```bash
# 使用镜像源（可选）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 2. FFmpeg 相关

**问题**: 音频/视频加载失败

**解决方案**:
```bash
# Conda 环境
conda install -c conda-forge ffmpeg

# 或通过 Homebrew
brew install ffmpeg

# 验证
ffmpeg -version
```

**问题**: OpenCV 找不到视频编解码器

**解决方案**:
```bash
# 重新安装 OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless
```

### 3. 依赖冲突

**问题**: 包版本冲突

**解决方案**:
```bash
# 完全重建环境
conda deactivate
conda env remove -n mmalign
conda env create -f environment.yml
conda activate mmalign
```

### 4. 内存问题

**问题**: 处理大视频时内存不足

**解决方案**:
- 降低目标采样率（如 `target_rate_hz=10` 而非 50）
- 分段处理视频
- 使用简单特征而非深度学习编码器

### 5. Streamlit 端口占用

**问题**: 端口 8501 已被占用

**解决方案**:
```bash
# 指定其他端口
streamlit run streamlit_app.py --server.port 8502
```

### 6. 模块导入错误

**问题**: `ModuleNotFoundError` 尽管已安装

**解决方案**:
```bash
# 确认使用正确的 Python
which python
# 应显示 .venv 或 conda 环境路径

# 检查已安装包
pip list | grep torch
pip list | grep streamlit
```

## GPU 加速（可选）

### Apple Silicon (M1/M2/M3)

PyTorch 自动支持 Metal Performance Shaders (MPS)：

```python
import torch

# 检查 MPS 可用性
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("使用 CPU")
```

### NVIDIA GPU (外接显卡)

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

## 更新与维护

### 更新 Conda 环境

```bash
conda activate mmalign
conda update --all
```

### 更新 pip 包

```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### 更新 PyTorch

```bash
# Apple Silicon
conda install pytorch torchvision torchaudio -c pytorch

# Intel Mac
pip install --upgrade torch torchvision torchaudio
```

## 卸载

### 删除 Conda 环境

```bash
conda deactivate
conda env remove -n mmalign
```

### 删除 venv 环境

```bash
deactivate
rm -rf .venv
```

### 删除缓存

```bash
# 删除 Jupyter 缓存
rm -rf notebooks/.ipynb_checkpoints

# 删除 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +

# 删除模型缓存（可选）
rm -rf ~/.cache/torch
rm -rf ~/.cache/huggingface
```

## 性能优化建议

### 针对 Apple Silicon

1. **使用 MPS 加速**:
```python
# 在编码器中指定 device
from mmalign.encoders import BERTTextEncoder
encoder = BERTTextEncoder(device="mps")
```

2. **优化内存使用**:
```bash
# 限制 PyTorch 线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

3. **使用优化的 NumPy**:
```bash
# Conda 会自动安装 Apple 优化的 Accelerate 框架
conda install numpy scipy -c conda-forge
```

### 通用优化

1. **批处理大文件**: 分段处理而非一次性加载
2. **降低采样率**: 对齐时使用 10-20Hz 而非 100Hz
3. **使用简单特征**: 快速原型阶段不使用深度学习
4. **缓存编码器**: 重用已加载的模型实例

## 下一步

安装完成后：

1. ✅ 运行 `streamlit run streamlit_app.py` 启动 Web 界面
2. ✅ 生成示例数据: `python scripts/generate_sample_data.py`
3. ✅ 在 Web 界面上传示例数据并尝试对齐
4. ✅ 浏览 `notebooks/` 中的 Jupyter 演示
5. ✅ 阅读 `README.md` 了解详细功能

## 技术支持

遇到问题：
1. 查看本文档的常见问题部分
2. 检查错误信息中的提示
3. 验证所有依赖已正确安装
4. 通过 GitHub Issues 反馈

---

**macOS 版本测试**:
- ✅ macOS 13 (Ventura) - Apple Silicon
- ✅ macOS 12 (Monterey) - Intel & Apple Silicon
- ✅ macOS 11 (Big Sur) - Intel & Apple Silicon
