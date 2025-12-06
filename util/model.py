import torch
import os

from transformers import ViTConfig, ViTForImageClassification

from model.v1_cola_model import ColaViTConfig, ColaViTForImageClassification
from model.v2_cola_model import convert_vit_to_cola_m

def build_model(cfg):

    vit_config = ViTConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_channels=cfg.num_channels,
        num_labels=cfg.num_classes,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
    )

    if cfg.model_name == "vit":
        vit = ViTForImageClassification(vit_config)
        return vit
    elif cfg.model_name == "v1_cola":
        cola = ColaViTForImageClassification(
            config=ColaViTConfig(
                cola_rank_ratio=cfg.cola_rank_ratio,
                lr_act_type=cfg.cola_lr_act_type,
                **vit_config.to_dict(),  # Inherit all standard args
            )
        )
        return cola
    elif cfg.model_name == "v2_cola":
        v2_cola = ViTForImageClassification(vit_config)
        v2_cola = convert_vit_to_cola_m(v2_cola, rank_ratio=cfg.cola_rank_ratio)
        return v2_cola

    raise ValueError(f"bad model name, ")
