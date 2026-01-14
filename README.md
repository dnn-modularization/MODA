# MODA: DNN Modularization via Activation-Driven Training

**Artifact for ICSE'26 Research Paper**

Preprint: https://arxiv.org/abs/2411.01074

## Overview

MODA enables deep neural network modularization through activation-driven training. This artifact supports three main capabilities:

1. **Model Training**: Train standard and modular models with configurable modularization objectives
2. **Module Composition**: Extract class-specific modules and compose them for targeted sub-tasks
3. **Module Replacement**: Repair weak models by replacing components with modules from stronger models

## Project Structure

```
configs.py                    # Configuration and paths
model_trainer.py             # Main training script
model_modularizer.py         # Module extraction and composition
dataset_loader/              # Dataset loading utilities
models/                      # Model architectures (VGG, ResNet, MobileNet, LeNet)
exp_analysis/               # Analysis tools for FLOPs and modules
exp_repair/                 # Model repair experiments
paper_figures/              # Generated figures (RQ1, RQ2, RQ3)
raw_data/                   # Datasets/logs/checkpoints
```

## Requirements

- **OS**: Ubuntu 20.04 LTS
- **GPU**: NVIDIA GPU with >16 GB VRAM (e.g., T4/V100); >60 GB VRAM for ImageNet experiments (e.g., GH200)
- **Python**: 3.8+ via Conda environment (see Setup below)

## Setup

### Environment

1. Install [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2) for Ubuntu

2. Create and activate the conda environment:
```bash
$ conda env create -f environment.yml
$ conda activate MODA_env
```

### Datasets

**CIFAR-10, CIFAR-100, SVHN** (automatic download):
- Downloaded automatically to `raw_data/torchvision_datasets/` on first run

**ImageNet** (manual setup):
1. Download ImageNet from https://image-net.org/
2. Class lists provided in `raw_data/`:
   - ImageNet100R: `100_random_classes.txt`
   - ImageNet100D: `100_dog_breed_classes.txt`
3. Organize as follows:
```
raw_data/torchvision_datasets/imagenet/
├── train/
│   ├── n01484850/
│   │   ├── n01484850_10016.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01484850/
    └── ...
```
*Alternative*: Download pre-processed datasets from https://www.kaggle.com/datasets/lolame/imagenet-variants

## Usage

### 1. Training Models

**Standard model** (baseline):
```bash
$ python3 model_trainer.py --model vgg16 --dataset cifar10 \
    --batch_size 128 --learning_rate 0.05 --n_epochs 200 \
    --checkpoint_every_n_epochs -1 \
    --wf_affinity 0.0 --wf_dispersion 0.0 --wf_compactness 0.0
```

**Modular model** (with MODA):
```bash
$ python3 model_trainer.py --model vgg16 --dataset cifar10 \
    --batch_size 128 --learning_rate 0.05 --n_epochs 200 \
    --checkpoint_every_n_epochs -1 \
    --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3
```

Parameters:
- `model`: {vgg16, resnet18, mobilenet}
- `dataset`: {svhn, cifar10, cifar100}
- `wf_affinity` (α), `wf_dispersion` (β), `wf_compactness` (γ): Modularization objective weights

### 2. Module Composition

Extract and compose class-specific modules:
```bash
$ python3 model_modularizer.py --model vgg16 --dataset cifar10 \
    --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3 \
    --activation_rate_threshold 0.9
```

Parameters:
- `activation_rate_threshold`: Threshold ∈ [0, 1] for module extraction (default: 0.9)

**Analysis tools**:

FLOPs comparison:
```bash
$ cd exp_analysis
$ python3 compare_flops.py --model vgg16 --dataset cifar100 \
    --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3 \
    --activation_rate_threshold 0.9
```

Module analysis:
```bash
$ python3 module_analysis.py --model vgg16 --dataset cifar10
```

### 3. Module Replacement (Model Repair)

Replace weak model components with modules from strong models:

**Train weak model**:
```bash
$ cd exp_repair
$ python3 weak_model_trainer.py --model lenet5 --dataset mixed_cifar10_for_repair \
    --batch_size 128 --learning_rate 0.05 --n_epochs 200 \
    --checkpoint_every_n_epochs 5
```

**Train strong model**:
```bash
$ python3 strong_model_trainer.py --model vgg16 --dataset mixed_cifar10_for_repair \
    --batch_size 128 --learning_rate 0.05 --n_epochs 200 \
    --checkpoint_every_n_epochs -1
```

**Perform replacement**:

Underfitted model repair (early checkpoint, e.g., epoch 10):
```bash
$ python3 weak_model_repair.py --weak_model lenet5 --strong_model vgg16 \
    --dataset mixed_cifar10_for_repair --mixed_class 0 \
    --repair_strategy moda --batch_size 128 --target_epoch 10
```

Overfitted model repair (late checkpoint, e.g., epoch 195):
```bash
$ python3 weak_model_repair.py --weak_model lenet5 --strong_model vgg16 \
    --dataset mixed_cifar10_for_repair --mixed_class 0 \
    --repair_strategy moda --batch_size 128 --target_epoch 195
```

Parameters:
- `weak_model`: {lenet5}
- `strong_model`: {vgg16, resnet18}
- `dataset`: {mixed_svhn_for_repair, mixed_cifar10_for_repair}
- `mixed_class`: {0, 1, 2, 3, 4}
- `repair_strategy`: {moda, cnnsplitter}
- `target_epoch`: Training checkpoint to repair [1, 200]

**Baselines**

Forked repositories for comparison with related approaches:

- **MwT**: https://github.com/dnn-modularization/forked_MwT
- **GradSplitter**: https://github.com/dnn-modularization/forked_GradSplitter
- **INCITE**: https://github.com/dnn-modularization/forked_INCITE

**Citation**

If you use this artifact, please cite:
```bibtex
@inproceedings{moda2026,
  title={DNN Modularization via Activation-Driven Training},
  author={Ngo, Tuan and Hassan, Abid and Shafiq, Saad and Medvidovic, Nenad},
  booktitle={Proceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering},
  series={ICSE '26},
  year={2026},
  location={Rio de Janeiro, Brazil},
  publisher={Association for Computing Machinery},
  address={New York, NY, USA},
  doi={10.1145/3744916.3773190},
  isbn={979-8-4007-2025-3}
}
```
