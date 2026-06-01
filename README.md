# Deep Variational Fusion: A New Framework to Learn Energy-Driven Distribution Mapping for Zero-Shot Pansharpening

> 在已有 Pansharpening 方法输出的初始 HRMS 结果基础上，对每张测试样本做**样本级两阶段交替优化**，结合 MTF 退化模型与可学习权重网络，进一步提升光谱保真度与空间细节。

---

## 📁 项目结构

```
DVF/
├── src/                  # 核心源码
│   ├── model.py          # 网络架构（NewNet, FusionNet, UNet）
│   ├── psutils.py        # MTF退化模型与图像处理工具
│   ├── metrics.py        # 评估指标（PSNR, SSIM, SAM, ERGAS等）
│   └── utils_log.py      # 日志工具
├── train/                # 训练脚本
│   ├── reduce_train_simple_v2.py   # 降分辨率训练（简化版）
│   └── full_train_simple_v2.py     # 全分辨率训练（简化版）
├── configs/              # 配置文件
│   ├── config_reduce_simple.yaml
│   └── config_full_simple.yaml
└── README.md
```

---

## 🚀 快速开始

### 降分辨率模式

```bash
# 使用命令行参数
python train/reduce_train_simple_v2.py \
    --sensor WV3 \

# 使用配置文件
python train/reduce_train_simple_v2.py --config configs/config_reduce_simple.yaml
```

### 全分辨率模式

```bash
# 使用命令行参数
python train/full_train_simple_v2.py \
    --sensor WV3 \

# 使用配置文件
python train/full_train_simple_v2.py --config configs/config_full_simple.yaml
```

---

## 🎯 核心算法

**两阶段交替优化：**

1. **初始化阶段**：固定 Xnet，训练权重网络 wnet
2. **交替优化阶段**：交替更新图像变量 X 和网络参数 θ

**损失函数：**
```
L = ||down(X_blur(X)) - Tlrms||² + λ₁||X - w·Xnet||² + λ₂·MSE(pred_pan, P3D)
```

---

## 📦 依赖

- PyTorch, torchvision
- numpy, scipy, h5py
- matplotlib
- pyyaml

---