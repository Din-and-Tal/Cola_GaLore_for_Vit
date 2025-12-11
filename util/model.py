from model.cola_model import convert_vit_to_cola_m
from transformers import ViTConfig, ViTForImageClassification

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

    elif cfg.model_name == "cola":
        cola = ViTForImageClassification(vit_config)
        cola = convert_vit_to_cola_m(
            model=cola,
            use_intermediate_rank_scale=cfg.use_intermediate_rank_scale,
            intermediate_size=cfg.intermediate_size,
            rank_ratio=cfg.cola_rank_ratio,
            lr_act_type=cfg.lr_act_type
        )
        return cola

    raise ValueError("bad model name, ")
