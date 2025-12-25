import os
import sys
from contextlib import redirect_stdout

import hydra
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.cola_model import convert_vit_to_cola_m


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
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=intermediate_size,
        layer_norm_eps=1e-6,
        qkv_bias=True,
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
    )

    # Create outputs directory
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Save original model summary and representation
    with open(os.path.join(outputs_dir, "vit_layers.txt"), "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            summary(
                model, input_size=(1, num_channels, image_size, image_size), depth=200
            )

    with open(os.path.join(outputs_dir, "vit_model.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))

    # Save cola summary and representation
    with open(os.path.join(outputs_dir, "cola_layers.txt"), "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            summary(
                cola, input_size=(1, num_channels, image_size, image_size), depth=50
            )

    with open(os.path.join(outputs_dir, "cola_model.txt"), "w", encoding="utf-8") as f:
        f.write(str(cola))
    print(f"Model outputs saved to {outputs_dir}")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # print(model)
    for module_name, module in model.named_modules():
        print(module_name, module)


if __name__ == "__main__":
    main()
