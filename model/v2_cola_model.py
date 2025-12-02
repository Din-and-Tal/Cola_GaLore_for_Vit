import torch
import torch.nn as nn

from model.v2_cola_layer import ColaLinear

def convert_vit_to_cola_m(model, rank_ratio=0.25, verbose=False):
    """
    Recursively replaces nn.Linear layers in a ViT model with ColaLinear layers.
    
    Args:
        model: HuggingFace ViTForImageClassification model
        rank_ratio: Ratio of rank to input dimension (default 1/4 per paper)
    """
    
    # We define a recursive function to traverse the model
    def replace_linear_with_cola(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # Identify target Linear layers
            # In ViT, we typically want to target:
            # - Attention: query, key, value, output.dense
            # - MLP: intermediate.dense, output.dense
            if isinstance(child, nn.Linear):
                
                # Check if it's the classifier head. 
                # Usually we KEEP the final classifier head full-rank for stability,
                # unless you specifically want to compress it too.
                if "classifier" in full_name or "head" in full_name:
                    if verbose: print(f"Skipping classifier: {full_name}")
                    continue

                # Calculate Rank per paper (r = d/4)
                in_feat = child.in_features
                out_feat = child.out_features
                
                # Use the smaller dimension to calculate rank to ensure compression
                base_dim = min(in_feat, out_feat)
                rank = max(1, int(base_dim * rank_ratio))
                
                if verbose:
                    print(f"Replacing {full_name} | In: {in_feat}, Out: {out_feat} -> CoLA Rank: {rank}")
                
                # Create CoLA Layer
                cola_layer = ColaLinear(
                    in_features=in_feat,
                    out_features=out_feat,
                    rank=rank,
                    bias=(child.bias is not None),
                    use_checkpointing=True # Enable CoLA-M behavior
                )
                
                # Replace in parent module
                setattr(module, name, cola_layer)
                
            else:
                # Recurse deeper
                replace_linear_with_cola(child, full_name)

    # print(f"Converting ViT to CoLA-M (Rank Ratio: {rank_ratio})...")
    replace_linear_with_cola(model)
    return model

# --- Usage Example ---

# # 1. Load standard ViT
# model_id = "google/vit-base-patch16-224"
# vit_model = ViTForImageClassification.from_pretrained(model_id)

# # 2. Check original parameter count
# original_params = sum(p.numel() for p in vit_model.parameters())
# print(f"Original Params: {original_params / 1e6:.2f} M")

# # 3. Convert to CoLA-M
# # The paper recommends rank = 1/4 of dimension
# cola_vit_model = convert_vit_to_cola_m(vit_model, rank_ratio=0.25)

# # 4. Check new parameter count
# cola_params = sum(p.numel() for p in cola_vit_model.parameters())
# print(f"CoLA Params: {cola_params / 1e6:.2f} M")

# # 5. Verify Structure
# print("\nFirst Encoder Layer Structure:")
# print(cola_vit_model.vit.encoder.layer[0])