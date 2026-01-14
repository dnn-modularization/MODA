import os
import sys
from pathlib import Path

# ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools
import random
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

from configs import ModelConfig
from dataset_loader import load_dataset
from model_modularizer import calculate_modular_layer_masks, compose_model_from_modular_masks
from models.model_utils import print_model_summary, powerset, count_parameters, get_runtime_device, \
    get_model_leaf_layers
from models.modular_utils import get_activation_rate_during_inference
from models import create_modular_model, compose_model_from_module_masks
from models.model_evaluation_utils import evaluate_model

DEVICE = get_runtime_device()


def evaluate_modules(model_type, raw_model, modular_masks_path, num_classes=10, input_size=(3, 32, 32)):
    model_param_count = count_parameters(raw_model)

    # all_classes__modular_layer_masks = torch.load(modular_masks_path)
    trackable_params = generate_trackable_params(raw_model.state_dict())
    modules = []
    module_total_sizes = []
    for curr_class in range(num_classes):
        curr_module = compose_model_from_modular_masks(model_type, trackable_params, modular_masks_path,
                                                       num_classes, [curr_class], input_size=input_size)
        modules.append(curr_module)
        curr_module_param_count = count_parameters(curr_module)
        module_total_sizes.append(curr_module_param_count / model_param_count)
        print(f"[Class {curr_class}] Module's param count: {curr_module_param_count:,} "
              f"({curr_module_param_count / model_param_count:.2f})")
    module_overlap_sizes = calculate_overlap_params(modules, model_param_count)
    # print(model_type)
    # print(raw_model)
    # print(modular_masks_path)
    # print(module_total_sizes)
    # print(module_overlap_sizes)
    print("module_total_size", sum(module_total_sizes) / len(module_total_sizes))
    print("module_overlap_sizes", sum(module_overlap_sizes) / len(module_overlap_sizes))


def generate_trackable_params(raw_model_params):
    # generate unique values to params to measure the overlap after those params are modularized to individual modules
    # this means the pretrained params will be replaced (so don't use it for evaluate accuracy of the model)

    trackable_model_params = {}
    unique_number = 0
    for param_name, params in raw_model_params.items():
        numel = params.numel()  # Total number of elements in the tensor
        new_tensor = torch.arange(unique_number, unique_number + numel, dtype=torch.float64)
        unique_number += numel
        trackable_model_params[param_name] = new_tensor.view(params.shape)
    
    all_flattened = torch.cat([p.flatten() for p in trackable_model_params.values()])
    unique_flattened = torch.unique(all_flattened)
    assert len(unique_flattened) == unique_number, (
        f"Expected {unique_number} unique values, got {len(unique_flattened)}"
    )

    return trackable_model_params


def calculate_overlap_params(modules, model_param_count):
    flatten_module_params = []
    for m in tqdm(modules, desc="Flattening module params"):
        flatten_param_list = []
        assert next(iter(m.float64_param.values())).dtype == torch.float64, \
            "m.float64_param must be float64 to keep the uniqueness of the ids, " \
            "float32 will give duplicate ids as its range is limited"
        flatten_param_list = torch.cat([p.view(-1) for p in m.float64_param.values()]).int().tolist()
        flatten_param_set = set(flatten_param_list)
        # Check uniqueness
        assert len(flatten_param_set) == len(flatten_param_list), f"Non-unique params in module {i}"
        flatten_module_params.append(flatten_param_set)

        del m

    # Calculate all combinations of modules
    indices_combinations = list(itertools.combinations(range(len(flatten_module_params)), 2))

    # 357 = sampling from population of [4950 indices_combinations] with confidence level of 95%, margin of error 5%
    if len(flatten_module_params) == 100:
        indices_combinations = random.sample(indices_combinations, k=357)

    print("indices_combinations", len(indices_combinations))
    overlap_sizes = []
    for module1_index, module2_index in tqdm(indices_combinations, desc="Calculating overlaps"):
        module1_params = flatten_module_params[module1_index]
        module2_params = flatten_module_params[module2_index]
        curr_intersection = len(module1_params & module2_params)
        # curr_union = len(module1_params | module2_params)
        overlap_sizes.append(curr_intersection / model_param_count)
    return overlap_sizes


def main():
    activation_rate_threshold = 0.9

    parser = argparse.ArgumentParser(description="Module Analysis")
    parser.add_argument('--model', type=str, default="vgg16", choices=["vgg16", "resnet18", "mobilenet"], help="Model architecture")
    parser.add_argument('--dataset', type=str, default="imagenet", choices=["svhn", "cifar10", "cifar100", "imagenet"], help="Dataset name")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset

    input_size, num_classes, _, _ = load_dataset(
        dataset_type=dataset_name,
        batch_size=128,
        num_workers=2
    )

    checkpoint_dir = f"{ModelConfig.model_checkpoint_dir}/{model_name}_{dataset_name}/"
    checkpoint_dir = os.path.join(
        checkpoint_dir,
        "model__bs128__ep200__lr0.05__aff1.0_dis1.0_comp0.3"
    )
    checkpoint_path = [
        entry.path for entry in os.scandir(checkpoint_dir)
        if entry.is_dir() and entry.name.startswith("v")
    ][-1] + "/model.pt"

    print(activation_rate_threshold, checkpoint_path)

    modular_masks_save_path = checkpoint_path + f".mod_mask.thres{activation_rate_threshold}.pt"

    raw_model = create_modular_model(model_type=model_name, num_classes=num_classes,
                                     input_size=input_size, modular_training_mode=True)

    evaluate_modules(model_type=model_name,
                     raw_model=raw_model,
                     modular_masks_path=modular_masks_save_path,
                     num_classes=num_classes,
                     input_size=input_size)


if __name__ == '__main__':
    main()
