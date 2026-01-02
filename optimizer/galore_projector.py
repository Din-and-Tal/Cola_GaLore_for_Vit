import torch


# GaLoreProjector: low-rank gradient projection for 2D tensors (matrices)
#
# Notation for complexity comments:
#   - Let full_rank_grad have shape (m, n)
#   - Let r = rank (self.rank), with r <= min(m, n)
#
# Costs (FLOP-ish):
#   - Matrix-matrix multiply (A: a×b, B: b×c):  O(a * b * c)
#   - SVD (m×n):  O(min(m, n) * m * n)   (Hessenberg + QR-ish)
#
# Example (ViT-Tiny MLP weight):
#   - Suppose m = 768, n = 192, r = 64
#   - full_rank_grad shape = (768, 192)
#   - One SVD:
#       O(min(768,192) * 768 * 192) ≈ O(192 * 768 * 192) ≈ 28M FLOPs
#   - One projection (768×192) @ (192×64):
#       O(768 * 192 * 64) ≈ 9.4M FLOPs
#   - One back-projection (768×64) @ (64×192):
#       O(768 * 64 * 192) ≈ 9.4M FLOPs
#
# Amortized SVD cost per step if update_proj_gap = G:
#   ≈ O( (min(m,n) * m * n) / G )
#
# So total overhead per step per matrix:
#   O( m * n * r   [project]  +  m * r * n  [project_back]  +  (min(m,n) * m * n) / G )
# ≈ O( 2 * m * n * r + (min(m,n) * m * n) / G )

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank 
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

        # update_proj_gap = G: recompute SVD every G steps
        #   - Smaller G: fresher projector, more SVD cost
        #   - Larger G: staler projector, less SVD cost

    def project(self, full_rank_grad, iter):
        """
        Complexity (per call, ignoring SVD recompute):
            - Main cost is a single matmul:
                (m, n) @ (n, r)  or  (r, m) @ (m, n)
              So: O(m * n * r)

        SVD recompute cost (when triggered):
            - get_orthogonal_matrix:
                O(min(m, n) * m * n)

        Amortized per step:
            - Total ≈ O(m * n * r)  +  O(min(m,n) * m * n / G)

        Example (ViT-Tiny, W_fc1: m=768, n=192, r=64, G=200):
            - project matmul:  ≈ 9.4M FLOPs
            - SVD:             ≈ 28M FLOPs
            - amortized SVD:   ≈ 0.14M FLOPs per step
            - total per step:  ≈ 9.5M FLOPs per parameter matrix
        """
        if self.proj_type == 'std': # O(m*n*r) per step  +  O(min(m,n)*m*n/G) amortized
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'reverse_std':  # O(m*n*r)  +  O(min(m,n)*m*n/G)
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':  # O(m*n*r) + O(min(m,n)*m*n/G)
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':  # O(m*n*r) + O(min(m,n)*m*n/G)
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':  # O(m*n*r) + O(min(m,n)*m*n/G)
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):

        """
         Map low-rank gradient back to approximate full-rank gradient.

         Complexity:
             - One matmul:
                 - Either (m, r) @ (r, n) or (m, r) @ (r, n)
               So: O(m * n * r)
             - For 'full': two matmuls:
                 A @ (r×r) @ B → still O(m * n * r) when m,n >> r

         Example (ViT-Tiny, m=768, n=192, r=64):
             - back matmul: ≈ 9.4M FLOPs
         """

        if self.proj_type == 'std': # O(m*n*r), same for all
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device.type)


        return full_rank_grad * self.scale


    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        """
        Complexity:
            - torch.linalg.svd(matrix, full_matrices=False):
                O(min(m, n) * m * n)

        Example (ViT-Tiny W_fc1, m=768, n=192):
            - min(m,n) = 192
            - Cost ≈ 192 * 768 * 192 ≈ 28M FLOPs
        """
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')
