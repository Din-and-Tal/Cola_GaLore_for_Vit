import os
import sys
from contextlib import redirect_stdout

import hydra
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.cola_model import convert_vit_to_cola_m


def print_nn_layers(module, prefix="", output_lines=None):
    """
    Recursively print wrapper classes (like ViTEmbeddings) with their nn layers inside.
    Shows both the structure (wrapper names) and the actual PyTorch layers.
    """
    if output_lines is None:
        output_lines = []

    # List of actual PyTorch layer types to print directly
    actual_layers = (
        nn.Linear,
        nn.Conv2d,
        nn.Conv1d,
        nn.Conv3d,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Dropout,
        nn.GELU,
        nn.ReLU,
        nn.Softmax,
        nn.Identity,
        nn.Embedding,
        nn.LSTM,
        nn.GRU,
        nn.RNN,
    )

    # Get all direct children that are nn.Modules
    children = list(module.named_children())

    # If this module has no children, check if it's an actual layer
    if len(children) == 0:
        if isinstance(module, actual_layers):
            output_lines.append(f"{prefix}{module}")
        return output_lines

    # Process children recursively
    for name, child in children:
        class_name = child.__class__.__name__

        # Check if child is an actual layer
        if isinstance(child, actual_layers):
            output_lines.append(f"{prefix}({name}): {child}")
        elif isinstance(child, nn.Module):
            child_children = list(child.named_children())

            if isinstance(child, nn.Sequential):
                # Sequential - print header, recurse with indentation, close
                output_lines.append(f"{prefix}({name}): Sequential(")
                for idx, submodule in enumerate(child):
                    if isinstance(submodule, actual_layers):
                        output_lines.append(f"{prefix}  ({idx}): {submodule}")
                    else:
                        output_lines.append(
                            f"{prefix}  ({idx}): {submodule.__class__.__name__}("
                        )
                        print_nn_layers(submodule, prefix + "    ", output_lines)
                        output_lines.append(f"{prefix}  )")
                output_lines.append(f"{prefix})")
            elif isinstance(child, nn.ModuleList):
                # ModuleList - print header, recurse with indentation, close
                output_lines.append(f"{prefix}({name}): ModuleList(")
                for idx, submodule in enumerate(child):
                    if isinstance(submodule, actual_layers):
                        output_lines.append(f"{prefix}  ({idx}): {submodule}")
                    else:
                        output_lines.append(
                            f"{prefix}  ({idx}): {submodule.__class__.__name__}("
                        )
                        print_nn_layers(submodule, prefix + "    ", output_lines)
                        output_lines.append(f"{prefix}  )")
                output_lines.append(f"{prefix})")
            elif len(child_children) > 0:
                # Container with children - print wrapper name, recurse, close
                output_lines.append(f"{prefix}({name}): {class_name}(")
                print_nn_layers(child, prefix + "  ", output_lines)
                output_lines.append(f"{prefix})")
            else:
                # Leaf module that's not in our list - print it
                output_lines.append(f"{prefix}({name}): {child}")

    return output_lines


@hydra.main(version_base=None, config_path="../conf", config_name="vit_adamw")
def main(cfg):
    # Build ViT model using Hugging Face transformers to mirror model.py style
    from transformers import ViTConfig, ViTForImageClassification

    image_size = cfg.image_size
    patch_size = cfg.patch_size
    num_channels = cfg.num_channels
    num_labels = cfg.num_classes

    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size

    config = ViTConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_channels=cfg.num_channels,
        num_labels=cfg.num_classes,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=cfg.intermediate_size,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
    )
    model = ViTForImageClassification(config)

    from torchinfo import summary

    # cfg.cola_use_intermediate_rank_scale = True

    cola = ViTForImageClassification(config)
    cola = convert_vit_to_cola_m(
        model=cola,
        cola_use_intermediate_rank_scale=cfg.cola_use_intermediate_rank_scale,
        intermediate_size=cfg.intermediate_size,
        rank_ratio=cfg.cola_rank_ratio,
        cola_act=cfg.cola_act,
    )

    # Create outputs directory
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Use a for loop to reduce code duplication
    models = [
        ("vit", model, 200),
        ("cola", cola, 200),
    ]

    for model_name, model_obj, depth in models:
        # Save model summary
        with open(
            os.path.join(outputs_dir, f"{model_name}_layers.txt"), "w", encoding="utf-8"
        ) as f:
            with redirect_stdout(f):
                summary(
                    model_obj,
                    input_size=(1, num_channels, image_size, image_size),
                    depth=depth,
                )

        # Save model representation with only nn layers
        with open(
            os.path.join(outputs_dir, f"{model_name}_model.txt"), "w", encoding="utf-8"
        ) as f:
            output_lines = print_nn_layers(model_obj)
            f.write("\n".join(output_lines))

    print(f"Model outputs saved to {outputs_dir}")


if __name__ == "__main__":
    main()
