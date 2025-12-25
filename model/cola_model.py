import torch.nn as nn
from transformers.activations import ACT2FN

from model.cola_layer import ColaLinear


def convert_vit_to_cola_m(
    model,
    cola_use_intermediate_rank_scale,
    intermediate_size,
    rank_ratio=0.25,
    cola_act="silu",
    cola_use_checkpointing=True,
):
    lr_act = ACT2FN[cola_act]

    def replace_linear_with_cola(
        module,
        name_prefix="",
    ):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(child, nn.Linear):
                if "classifier" in full_name or "head" in full_name:
                    continue

                in_feat = child.in_features
                out_feat = child.out_features

                base_dim = min(in_feat, out_feat)

                if (
                    cola_use_intermediate_rank_scale
                    and "intermediate.dense" in full_name
                ):
                    base_dim = intermediate_size

                rank = max(1, int(base_dim * rank_ratio))

                cola_layer = ColaLinear(
                    in_features=in_feat,
                    out_features=out_feat,
                    rank=rank,
                    bias=(child.bias is not None),
                    cola_use_checkpointing=cola_use_checkpointing,
                    cola_act=cola_act,
                )

                setattr(module, name, cola_layer)

            elif child.__class__.__name__ == "GELUActivation":
                setattr(module, name, lr_act)

            else:
                replace_linear_with_cola(module=child, name_prefix=full_name)

    replace_linear_with_cola(module=model)
    return model
