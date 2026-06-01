# DVF - Deep Visual Fusion for Pansharpening

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

### 降分辨率模式（有GT）

```bash
# 使用命令行参数
python train/reduce_train_simple_v2.py \
    --sensor WV3 \
    --init_model DCFNet \
    --lam1 0.06 \
    --lam2 0.94

# 使用配置文件
python train/reduce_train_simple_v2.py --config configs/config_reduce_simple.yaml
```

### 全分辨率模式（无GT）

```bash
# 使用命令行参数
python train/full_train_simple_v2.py \
    --sensor WV3 \
    --init_model ADWM \
    --lam1 0.187 \
    --lam2 0.383

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

## 📝 提交记录

| Commit | 描述 |
|--------|------|
| `afa0b89` | 添加配置文件支持 |
| `103199f` | 添加核心源码和简化训练脚本 |

---

**Author**: Yinsen  
**GitHub**: [https://github.com/Y-sen-U/DVF](https://github.com/Y-sen-U/DVF)