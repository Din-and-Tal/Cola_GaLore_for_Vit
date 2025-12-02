import os
import torch
import random
import numpy as np


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # will increase library footprint in GPU memory by approximately 24MiB

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

        
