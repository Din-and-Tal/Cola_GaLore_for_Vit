from transformers import ViTConfig, ViTForImageClassification

from model.cola_model import convert_vit_to_cola_m


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

    model = None
    if cfg.use_cola:
        model = ViTForImageClassification(vit_config)
        model = convert_vit_to_cola_m(
            model=model,
            cola_use_intermediate_rank_scale=cfg.cola_use_intermediate_rank_scale,
            intermediate_size=cfg.intermediate_size,
            rank_ratio=cfg.cola_rank_ratio,
            cola_act=cfg.cola_act,
        )
    else:
        model = ViTForImageClassification(vit_config)
    
    if cfg.use_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    return model
