# MODA: DNN Modularization via Activation-Driven Training

### Experiment Setup:
- Ubuntu 20.04
- CUDA 10.2 & cuDNN 7.6.5
- Python 3.8.10
- Pip3 dependencies from `./requirements.txt`

### Data Preparation:

**Automatic Download** (CIFAR-10, CIFAR-100, SVHN):
- These datasets will be downloaded automatically to `./MODA/raw_data/torchvision_datasets/` during first run

**Manual Download** (ImageNet):
- Download the ImageNet dataset from https://image-net.org/. The selected classes for ImageNet100R are listed in `./MODA/raw_data/100_random_classes.txt`, and those for ImageNet100D are in `./MODA/raw_data/100_dog_breed_classes.txt`.
- Extract and organize the ImageNet dataset with the following folder structure (note: each class label is the folder name itself):
```
./MODA/raw_data/torchvision_datasets/imagenet/
├── train/
│   ├── n01484850/
│   │   ├── n01484850_10016.JPEG
│   │   ├── n01484850_18018.JPEG
│   │   └── ...
│   ├── n01530575/
│   ├── n01833805/
│   └── ... (other class folders)
└── val/
    ├── n01484850/
    ├── n01530575/
    ├── n01833805/
    └── ... (other class folders)
```

### Run steps:


**1. Training target model:**

1.1. Train standard model:
```sh
$ python3 model_trainer.py --model vgg16 --dataset cifar10 --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1 --wf_affinity 0.0 --wf_dispersion 0.0 --wf_compactness 0.0
```

1.2. Train modular model:
```sh
$ python3 model_trainer.py --model vgg16 --dataset cifar10 --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1 --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3
```

Arguments: 
- `model`={vgg16, resnet18, mobilenet}
- `dataset`={svhn, cifar10, cifar100}
- `wf_affinity` (alpha), `wf_dispersion` (beta), `wf_compactness` (gamma)

**2. Module Composition for Sub-tasks:**

> Extract class-specific modules and compose them for target classes.

**Key Steps:**
1. **Decomposition**: Calculate class-specific activation rates during inference with `activation_rate_threshold`, each class has minimum 0.5% neurons per layer.
2. **Composition**:
   - **VGG (Sequential CNN)**: 
     * **Constraints**: Input/output channel matching between consecutive layers
     * Sequential parameter slicing: `params[:, prev_mask]` for input dim, `params[curr_mask]` for output dim
     * Conv-to-FC transition handled by (C×H×W flattening) as during forward()
   
   - **ResNet (Skip Connections)**:
     * **Constraints**: Skip connection dimension compatibility; shared activations between conv2 and downsample layers
     * Downsample layers and main conv share the same channel mask from post-activation
     * When downsample doesn't exist, forward pass uses indexed selection: `out[:, conv2_indices] += identity[:, identity_indices]`
   
   - **MobileNet (Depthwise Separable)**:
     * **Constraints**: Depthwise structure requires group convolution (in_channels = groups = out_channels)
     * Only retain subsequent conv's channels in preceding conv's channels (1-1 mapping in depthwise)
     * Missing channels in preceding conv compared to subsequent conv are padded with zeros

> Edit model_checkpoint_dir in model_modularizer.py to point to the directory containing modular model trained in Step 1

```sh
$ python3 model_modularizer.py --model vgg16 --dataset cifar10 --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3 --activation_rate_threshold 0.9
```

Arguments: 
- `model`={vgg16, resnet18, mobilenet}
- `dataset`={svhn, cifar10, cifar100}
- `activation_rate_threshold`=[0, 1]

**3. Module Replacement for Model Repair:**

> Replace components of weak models (LeNet5) with modules from strong models (VGG16/ResNet18) to improve accuracy on specific classes. Uses modular extraction and calibration layers for compatibility.

3.1. Change base directory:
```sh
$ cd exp_repair
```
3.2. Train weak model:
```sh
$ python3 weak_model_trainer.py --model lenet5 --dataset mixed_cifar10_for_repair --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1
```

Arguments: 
- `model`={lenet5}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}


3.3. Train strong model:
```sh
$ python3 strong_model_trainer.py --model vgg16 --dataset mixed_cifar10_for_repair --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1
```

Arguments: 
- `model`={vgg16, resnet18}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}

3.4. Perform module replacement:

**Key Steps:**
1. **Module Extraction**: Extract class-specific module from strong model (VGG16/ResNet18) using same decomposition process as section 2
2. **Replacement Strategies**:
   - **MODA Strategy**: 
        * Freeze both weak model and strong module parameters
        * Replace weak model's target class output with strong module output: `concat(weak_output[non_target], strong_output)`
        * Train calibration layer (Linear) to align combined outputs for a selected number of epochs

   - **CNNSplitter Strategy**:
        * Normalize both weak model outputs with softmax.
        * Normalize strong module outputs to [0,1] range using min-max scaling.
        * Perform direct output replacement.

```sh
$ python3 weak_model_repair.py --weak_model lenet5 --strong_model vgg16 --dataset mixed_cifar10_for_repair --mixed_class 0 --repair_strategy moda --batch_size 128 --target_epoch 5
```

Arguments: 
- `weak_model`={lenet5}
- `strong_model`={vgg16, resnet18}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}
- `mixed_class`={0,1,2,3,4}
- `repair_strategy`={moda, cnnsplitter}
- `target_epoch`=[1, 200]


> To compare MODA with MwT, find the MwT's source code here https://github.com/dnn-modularization/forked_MwT
