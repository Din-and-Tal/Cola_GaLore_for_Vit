import os
import sys

import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.v2_cola_model import convert_vit_to_cola_m

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
        hidden_dropout_prob=getattr(cfg, "hidden_dropout_prob", 0.0),
    )
    model = ViTForImageClassification(config)

    from torchinfo import summary

    # summary(model, input_size=(1, num_channels, image_size, image_size), depth=200)
    cfg.intermediate_rank_scale = True

    print(model)
    v2_cola = ViTForImageClassification(config)
    v2_cola = convert_vit_to_cola_m(
        model=v2_cola,
        intermediate_rank_scale=cfg.intermediate_rank_scale,
        intermediate_size=cfg.intermediate_size,
    )
    summary(v2_cola, input_size=(1, num_channels, image_size, image_size), depth=50)
    # print("\n\n\n\n")
    # print(v2_cola)


if __name__ == "__main__":
    main()
