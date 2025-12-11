import torch.nn as nn

from model.cola_layer import ColaLinear
from transformers.activations import ACT2FN


def convert_vit_to_cola_m(
    model,
    use_intermediate_rank_scale,
    intermediate_size,
    rank_ratio=0.25,
    lr_act_type="silu"
):
    lr_act = ACT2FN[lr_act_type]
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

                if use_intermediate_rank_scale and "intermediate.dense" in full_name:
                    base_dim = intermediate_size

                rank = max(1, int(base_dim * rank_ratio))

                cola_layer = ColaLinear(
                    in_features=in_feat,
                    out_features=out_feat,
                    rank=rank,
                    bias=(child.bias is not None),
                    use_checkpointing=True,
                    lr_act_type=lr_act_type
                )

                setattr(module, name, cola_layer)

            elif child.__class__.__name__ == "GELUActivation":
                setattr(module, name, lr_act)

            else:
                replace_linear_with_cola(module=child, name_prefix=full_name)

    replace_linear_with_cola(module=model)
    return model
