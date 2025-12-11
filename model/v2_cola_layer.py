import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN


class ColaMDownProjLayer(nn.Module):
    def __init__(
        self, in_features, out_features, rank, lr_act=True, lr_act_type="silu"
    ):
        super().__init__()

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        target_sdv = (in_features + out_features) ** (-0.5)
        self.cola_a = nn.Parameter(
            torch.randn(in_features, rank) / (rank**0.25) * (target_sdv**0.5)
        )

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)
        if hasattr(self, "lr_act"):
            out = self.lr_act(out)
        return out


class ColaMUpProjLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_b = nn.Parameter(
            torch.randn(rank, out_features) / (rank**0.25) * (target_sdv**0.5)
        )

        if bias:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = torch.matmul(x, self.cola_b)
        if self.bias is not None:
            out += self.bias
        return out


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
        use_checkpointing=True,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.down = ColaMDownProjLayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            lr_act=True,
            lr_act_type=lr_act_type,
        )

        self.up = ColaMUpProjLayer(
            in_features=in_features, out_features=out_features, rank=rank, bias=bias
        )

    def forward(self, x):

        low_rank_act = self.down(x)

        if self.training and self.use_checkpointing and x.requires_grad:
            output = checkpoint(self.up, low_rank_act, use_reentrant=True,preserve_rng_state =False)
        else:
            output = self.up(low_rank_act)

        return output
