import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

# --- Paste your provided classes here (ColaMDownProjLayer, ColaMUpProjLayer) ---
# I have slightly modified them to ensure they pass arguments correctly within the wrapper

class ColaMDownProjLayer(nn.Module):
    def __init__(self, in_features, rank, lr_act=True, lr_act_type="silu"):
        super().__init__()
        self.rank = rank
        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]
        
        # Initialization based on paper's logic to maintain variance
        # "target_sdv" acts as a scaling factor
        self.cola_a = nn.Parameter(torch.randn(in_features, rank) / (rank ** 0.25))

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)
        if hasattr(self, "lr_act"):
            out = self.lr_act(out)
        return out

class ColaMUpProjLayer(nn.Module):
    def __init__(self, out_features, rank, bias=True):
        super().__init__()
        self.cola_b = nn.Parameter(torch.randn(rank, out_features) / (rank ** 0.25))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = torch.matmul(x, self.cola_b)
        if self.bias is not None:
            out += self.bias
        return out

# --- The Main CoLA-M Wrapper ---

class ColaLinear(nn.Module):
    """
    Replaces a standard nn.Linear layer with the CoLA-M Auto-Encoder architecture.
    Formula: h = B * sigma(A * x)
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        rank, 
        bias=True, 
        lr_act_type="silu",
        use_checkpointing=True
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Down Projection (A) + Activation
        self.down = ColaMDownProjLayer(
            in_features=in_features, 
            rank=rank, 
            lr_act=True, 
            lr_act_type=lr_act_type
        )
        
        # Up Projection (B) + Bias
        self.up = ColaMUpProjLayer(
            out_features=out_features, 
            rank=rank, 
            bias=bias
        )

    def forward(self, x):
        # 1. Compute Low-Rank Activations (The "Bottleneck")
        # According to the paper, this is the cached activation for CoLA-M
        low_rank_act = self.down(x)

        # 2. Compute Up-Projection
        # CoLA-M: Use checkpointing here. 
        # We drop the intermediate graph of the 'up' layer to save memory
        # and recompute it during backward pass.
        if self.training and self.use_checkpointing and x.requires_grad:
            output = checkpoint(self.up, low_rank_act, use_reentrant=False)
        else:
            output = self.up(low_rank_act)
            
        return output