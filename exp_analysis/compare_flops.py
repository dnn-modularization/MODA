import os
import sys
from pathlib import Path

# ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fvcore.nn import FlopCountAnalysis
from configs import ModelConfig, BaseConfig

from dataset_loader import load_dataset
from model_modularizer import (
    get_args,
    calculate_modular_layer_masks,
    generate_model_composition_tasks,
    modularize_and_compose_model,
    evaluate_modularization_performance,
    DEVICE
)
from models import create_modular_model
from models.model_utils import print_model_summary, count_parameters


def main():
    batch_size = 128
    args = get_args()
    print(args.__dict__)

    activation_rate_threshold = args.activation_rate_threshold
    dataset_type = args.dataset
    model_type = args.model

    model_checkpoint_dir = f"{ModelConfig.model_checkpoint_dir}/{model_type}_{dataset_type}/"

    mod_checkpoint_dir = os.path.join(
        model_checkpoint_dir,
        f"model__bs128__ep200__lr0.05__aff{args.wf_affinity}"
        f"_dis{args.wf_dispersion}_comp{args.wf_compactness}"
    )
    mod_checkpoint_path = [
        entry.path for entry in os.scandir(mod_checkpoint_dir)
        if entry.is_dir() and entry.name.startswith("v")
    ][-1] + "/model.pt"

    raw_checkpoint_dir = os.path.join(
        model_checkpoint_dir,
        "model__bs128__ep200__lr0.05__aff0.0_dis0.0_comp0.0"
    )
    raw_checkpoint_path = [
        entry.path for entry in os.scandir(raw_checkpoint_dir)
        if entry.is_dir() and entry.name.startswith("v")
    ][-1] + "/model.pt"

    print(f"Raw model checkpoint path: {raw_checkpoint_path}")
    print(f"Modular model checkpoint path: {mod_checkpoint_path}")

    num_classes, train_loader, _ = load_dataset(
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=2
    )

    # Load and summarize the standard (non-modular) model
    std_model = create_modular_model(
        model_type=model_type,
        num_classes=num_classes,
        modular_training_mode=False
    )
    std_model.load_pretrained_weights(raw_checkpoint_path)
    std_model.cuda()
    print_model_summary(std_model)

    # Load the trained modular model
    mod_model = create_modular_model(
        model_type=model_type,
        num_classes=num_classes,
        modular_training_mode=True
    )
    mod_model.load_pretrained_weights(mod_checkpoint_path)

    # Compute or load the modular masks
    modular_masks_save_path = f"{mod_checkpoint_path}.mod_mask.thres{activation_rate_threshold}.pt"
    if not os.path.exists(modular_masks_save_path):
        calculate_modular_layer_masks(
            model=mod_model,
            data_loader=train_loader,
            num_classes=num_classes,
            save_path=modular_masks_save_path,
            activation_rate_threshold=activation_rate_threshold
        )

    # # random input matching CIFAR-style shape
    # sample_input = torch.randn(1, 3, 32, 32, device=DEVICE)

    # random input matching Imagenet-style shape
    sample_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    # total FLOPs for the standard model
    std_model.eval()
    std_total_flops = FlopCountAnalysis(std_model, sample_input) \
        .unsupported_ops_warnings(False) \
        .total()
    

    # For each composition task, build the composed model and compare total FLOPs
    for target_classes in generate_model_composition_tasks(num_classes=num_classes):
        composed_model = modularize_and_compose_model(
            model_type=model_type,
            modular_model=mod_model,
            modular_masks_path=modular_masks_save_path,
            orig_num_classes=num_classes,
            target_classes=target_classes
        )

        composed_model.eval()

        # total FLOPs for the composed model
        com_total_flops = FlopCountAnalysis(composed_model.cuda(), sample_input) \
            .unsupported_ops_warnings(False) \
            .total()

        ratio = com_total_flops / std_total_flops if std_total_flops > 0 else float('nan')
        print(
            f"COM_MODEL_FLOPS/STD_MODEL_FLOPS: "
            f"{com_total_flops}/{std_total_flops}="
            f"{ratio:.4f}  -------- {target_classes}"
        )


if __name__ == "__main__":
    main()
