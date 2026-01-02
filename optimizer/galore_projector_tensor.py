import torch
from tensorly.decomposition import tucker
from tensorly import tenalg


# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors of dimension >2.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html


# Assume gradient tensor x has shape (n1, n2, ..., nd)
# Let:
#   N = n1 * n2 * ... * nd          # total elements
#   d = number of modes (len(x.shape))
#   r = rank (same per mode)
#   G = update_proj_gap
#
# Rough FLOP costs:
#   - Tucker decomposition (get_orthogonal_matrix):
#       O(N * sum_i n_i)      # done every G steps → amortized O(N * sum_i n_i / G) per step
#   - transform (project):    O(d * r * N) per step
#   - inverse_transform:      O(d * r * N) per step
#   - So projector overhead per optimizer step ≈ O(2 * d * r * N + N * sum_i n_i / G)
#
# -------------------------------------------------------------------------
# Example: ViT-Tiny-like tensor (e.g. a "conv-like" weight or reshaped block)
# -------------------------------------------------------------------------
#   shape: (192, 192, 3, 3)
#     -> n1 = 192, n2 = 192, n3 = 3, n4 = 3
#     -> d = 4
#     -> N = 192 * 192 * 3 * 3 = 331,776
#     -> sum_i n_i = 192 + 192 + 3 + 3 = 390
#   choose:
#     r = 8        # low rank per mode
#     G = 200      # update_proj_gap
#
#   - One Tucker decomposition:
#       ~ N * sum_i n_i ≈ 331,776 * 390 ≈ 1.29e8 FLOPs
#     Amortized per step: ≈ 1.29e8 / 200 ≈ 6.5e5 FLOPs
#
#   - One transform() (project):
#       ~ d * r * N ≈ 4 * 8 * 331,776 ≈ 1.06e7 FLOPs
#
#   - One inverse_transform() (project_back):
#       ~ d * r * N ≈ 1.06e7 FLOPs
#
#   → Total projector overhead per step for THIS tensor:
#       ≈ 2 * 1.06e7 + 6.5e5 ≈ 2.18e7 FLOPs
#     For comparison, a plain AdamW-style elementwise update on this tensor
#     is O(N) ≈ 3.3e5–3.3e6 FLOPs, so GaLore tensor projection can be ~7–8×
#     more compute on this tensor, trading compute for memory savings.


class GaLoreProjectorTensor:
    """
    A class that represents a projector for the GaLore algorithm.

    Args:
        rank (int): The rank of the projector.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): The number of iterations between updating the orthogonal matrix. Defaults to 200.
        scale (float, optional): The scaling factor for the projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.


        Cost per call (for tensor x of size N, d modes, rank r):
            - Optional Tucker update:
                  O(N * sum_i n_i) when it happens,
                  amortized ≈ O(N * sum_i n_i / G) per step.
            - transform() call:
                  O(d * r * N) every step.

        ViT-Tiny example (shape (192,192,3,3), r=8, G=200):
            - Amortized Tucker cost per step ≈ 0.65M FLOPs
            - transform() ≈ 10.6M FLOPs
        """

        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)

        # Apply low-rank projection via multi_mode_dot:
        #   x_low = x ×_1 U1^T ×_2 U2^T × ... ×_d Ud^T
        # Cost ≈ O(d * r * N) FLOPs.

        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.


        Cost per call:
            - inverse_transform(): O(d * r * N)
            - final scaling: O(N)

        ViT-Tiny example (shape (192,192,3,3), r=8):
            - inverse_transform() ≈ 10.6M FLOPs
            - scaling ≈ 0.33M FLOPs
        """

        # Reconstruct approximate full-rank grad:
        #   x_full ≈ x_low ×_1 U1 ×_2 U2 × ... ×_d Ud
        # using the same factors learned by Tucker.

        full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes the orthogonal matrix using SVD decomposition.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank_all (int): The desired rank of the decomposition.

        Returns:
            tuple: A tuple containing the core and factors of the orthogonal matrix.


        Cost (one-time, not every step):
            - HOSVD-style Tucker ≈ O(N * sum_i n_i)
              where N = product of dimensions, sum_i n_i = sum of mode sizes.
            - If called every G steps, amortized per step ≈ O(N * sum_i n_i / G).

        ViT-Tiny example (shape (192,192,3,3), r=8, G=200):
            - One Tucker ≈ 1.29e8 FLOPs
            - Amortized per step ≈ 6.5e5 FLOPs
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float() 
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all) 
        return tucker_tensor

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.


        Cost:
            - multi_mode_dot(x, factors, transpose=True)
            - For d modes, rank r: ≈ O(d * r * N)

        ViT-Tiny example (shape (192,192,3,3), r=8):
            - ≈ 4 * 8 * 331,776 ≈ 1.06e7 FLOPs
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.

        Cost:
            - multi_mode_dot(x, factors)
            - For d modes, rank r: ≈ O(d * r * N)

        ViT-Tiny example (shape (192,192,3,3), r=8):
            - ≈ 1.06e7 FLOPs
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)
