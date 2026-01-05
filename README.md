# 🚀 CoLA + GaLore for Vision Transformers

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Memory-Efficient Pre-Training of Large Vision Transformers on Single GPU**

This project implements and combines two state-of-the-art rank decomposition methods: **CoLA** (Low-Rank Activation) and **GaLore** (Gradient Low-Rank Projection) to enable efficient pre-training of Vision Transformers (ViT) from scratch on a single GPU without significant performance degradation.

<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/136077fe-9518-48e6-9fee-6158de9e1677" />

## 🎯 Overview

Training large Vision Transformers from scratch requires substantial GPU memory, often limiting researchers and practitioners to smaller models or requiring expensive multi-GPU setups. This project addresses this challenge by:

1. **CoLA (Compute-Efficient Pre-Training via Low-Rank Activation)**: Decomposes model activations using low-rank matrices, reducing memory footprint during forward passes.
2. **GaLore (Gradient Low-Rank Projection)**: Projects gradients into low-rank subspaces using SVD decomposition, dramatically reducing optimizer memory requirements.

By combining both techniques, we achieve significant memory savings while maintaining competitive model performance on CIFAR-10 classification tasks.

## ✨ Key Features

- 🔬 **Dual Decomposition Strategy**: Combines activation (CoLA) and gradient (GaLore) decomposition
- 🎨 **Flexible Model Sizes**: Support for tiny, small, base, large, and huge ViT configurations
- 💾 **Memory Profiling**: Built-in memory measurement and breakdown analysis
- 📊 **WandB Integration**: Comprehensive experiment tracking and visualization
- ⚙️ **Multiple Optimizers**: Support for AdamW, GaLore-AdamW, and 8-bit variants
- 🔄 **Gradient Checkpointing**: Optional activation checkpointing for additional memory savings
- 🎯 **Hydra Configuration**: Flexible configuration management with Hydra

## 📦 Installation

### Prerequisites

- CUDA-capable GPU (tested on CUDA 13.0)
- Conda or Miniconda

### Build Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd GaLore_Cola_for_Vit_v3
```

2. Create and activate the conda environment:
```bash
conda env create -f env.yaml
conda activate cola_galore
```

The environment includes:
- Python 3.13
- PyTorch with CUDA 13.0 support
- Transformers, Datasets, WandB
- TensorLy (for tensor decomposition)
- All other required dependencies

## 🚀 Quick Start

### Basic Usage

Run training with a specific configuration and model size:

```bash
python main/main.py --config-name <config> size=<size>
```

### Available Configurations

- `vit_adamw`: Standard ViT with AdamW optimizer (baseline)
- `vit_galore_layer`: ViT with GaLore optimizer (gradient decomposition)
- `cola_adamw`: ViT with CoLA layers (activation decomposition) + AdamW
- `cola_galore_layer`: ViT with CoLA layers + GaLore optimizer (both decompositions)

### Available Model Sizes

- `tiny`: 192 hidden size, 12 layers, 3 attention heads
- `small`: 384 hidden size, 12 layers, 6 attention heads
- `base`: 768 hidden size, 12 layers, 12 attention heads
- `large`: 1024 hidden size, 24 layers, 16 attention heads
- `huge`: 1280 hidden size, 32 layers, 16 attention heads

### Example Commands

```bash
# Train a tiny ViT with CoLA + GaLore
python main/main.py --config-name cola_galore_layer size=tiny

# Train a large ViT with standard AdamW (baseline)
python main/main.py --config-name vit_adamw size=large

# Train with custom batch size and checkpointing
python main/main.py --config-name cola_galore_layer size=base batch_size=256 use_checkpointing=true
```

### Advanced Options

```bash
# Disable full training (quick test run)
python main/main.py --config-name cola_galore_layer size=tiny full_train=false

# Custom WandB project
python main/main.py --config-name cola_galore_layer size=base wandb_project_name=my_experiment

# Enable mixed precision training
python main/main.py --config-name cola_galore_layer size=base use_amp=true
```

## 📊 Results

### Performance Comparison

Our experiments compare four configurations across different model sizes and training settings:

1. **ViT + AdamW** (baseline)
2. **ViT + GaLore** (gradient decomposition only)
3. **CoLA + AdamW** (activation decomposition only)
4. **CoLA + GaLore** (both decompositions)

### Wandb Report

[Report](https://api.wandb.ai/links/din-alon-technion-israel-institute-of-technology/tar40u65)

<img width="1770" height="644" alt="image" src="https://github.com/user-attachments/assets/42e0c1d7-232f-4954-843f-2a1446a48a94" />
<img width="1770" height="644" alt="image" src="https://github.com/user-attachments/assets/2b21c624-e5f2-4a22-b639-cd8bedc94074" />
<img width="1770" height="644" alt="image" src="https://github.com/user-attachments/assets/2400dcd7-ebd4-4d99-9504-8233d9b1b026" />
<img width="1770" height="644" alt="image" src="https://github.com/user-attachments/assets/3d660ed0-28d8-41cd-9c2e-c37f9822df5a" />

### Key Findings

- **Memory Reduction**: ~80% memory savings with combined CoLA+GaLore
- **Performance**: Minimal accuracy degradation (<2%) compared to baseline but compensated with longer training.
- **Scalability**: Enables training of "huge" ViT models on single 24GB GPU

## 🏗️ File Structure

```
GaLore_Cola_for_Vit_v3/
├── main/
│   └── main.py              # Entry point with Hydra configuration
├── model/
│   ├── cola_layer.py        # CoLA low-rank layer implementations
│   └── cola_model.py        # ViT to CoLA conversion utilities
├── optimizer/
│   ├── galore_setup.py      # GaLore optimizer configuration
│   ├── galore_projector.py  # 2D gradient projection (SVD)
│   ├── galore_projector_tensor.py  # N-D tensor projection (Tucker)
│   └── galore8bit.py        # 8-bit GaLore optimizer variant
├── train/
│   ├── trainer_setup.py     # Trainer initialization
│   ├── trainer_loop.py      # Training loop implementation
│   └── trainer_utils.py     # Training utilities
├── util/
│   ├── model.py             # Model building utilities
│   ├── dataloader.py        # CIFAR-10 data loading
│   ├── memory_record.py     # Memory profiling tools
│   └── optuna_utils.py      # Hyperparameter optimization
├── conf/
│   ├── base.yaml            # Base configuration
│   ├── vit_adamw.yaml       # Standard ViT config
│   ├── vit_galore_layer.yaml
│   ├── cola_adamw.yaml
│   └── cola_galore_layer.yaml
└── experiments/
    └── mem_diff.sh          # Memory comparison scripts
```

## 🔬 Methodology

### CoLA: Low-Rank Activation Decomposition

CoLA replaces standard linear layers with low-rank auto-encoder architectures:

```
Standard: h = Wx
CoLA:     h = B · σ(A · x)
```

Where:
- `A` is a low-rank down-projection matrix (in_features × rank)
- `σ` is an activation function (GELU/SiLU)
- `B` is a low-rank up-projection matrix (rank × out_features)

This reduces activation memory during forward passes, especially in the intermediate MLP layers of transformers.

**Key Parameters:**
- `cola_rank_ratio`: Ratio of rank to base dimension (default: 0.25)
- `cola_use_intermediate_rank_scale`: Special scaling for intermediate layers
- `cola_act`: Activation function type

### GaLore: Gradient Low-Rank Projection

GaLore projects gradients into low-rank subspaces using SVD decomposition:

1. **Projection**: `g_low = P^T · g_full` (project full-rank gradient to low-rank)
2. **Optimizer Step**: Apply AdamW on low-rank gradient
3. **Back-Projection**: `g_full ≈ P · g_low` (reconstruct full-rank update)

The projection matrix `P` is periodically updated via SVD of the weight matrix, amortizing the SVD cost over multiple steps.

**Key Parameters:**
- `galore_rank`: Rank of gradient projection (default: 128)
- `galore_update_proj_gap`: Steps between SVD updates (default: 200)
- `galore_scale`: Scaling factor for projected gradients (default: 1.0)

### Combined Approach

When both methods are used together:
- **Forward pass**: CoLA reduces activation memory
- **Backward pass**: GaLore reduces gradient and optimizer state memory
- **Result**: Significant overall memory reduction enabling larger models on single GPUs

## 📚 Citation

```bibtex
@article{cola2025,
  title={Compute-Efficient Pre-Training of LLMs via Low-Rank Activation},
  author={Liu, Alvin and Wang, Zhengyang},
  journal={arXiv preprint arXiv:2502.10940},
  year={2025}
}

@article{galore2024,
  title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection},
  author={Zhao, Jiawei and Zhang, Zhenyu and Chen, Beidi and Wang, Zhangyang and Anandkumar, Anima and Tian, Yuandong},
  journal={arXiv preprint arXiv:2403.03507},
  year={2024}
}
```

## 🙏 Acknowledgments

- **CoLA Implementation**: Based on [alvin-zyl/CoLA](https://github.com/alvin-zyl/CoLA)
- **GaLore Implementation**: Based on [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore)
- **Base Model Hyper-Parameters**: Based on [kentaroy47/vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
- **Vision Transformer**: Built on Hugging Face Transformers library
- **Dataset**: CIFAR-10 from PyTorch datasets

## 📝 License

This project is open source and available under the MIT License.

---
