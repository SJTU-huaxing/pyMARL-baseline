import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

class Transform:
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def infer_output_info(self, vshape_in: tuple, dtype_in: torch.dtype) -> tuple:
        raise NotImplementedError
    
class OneHot(Transform):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.one_hot(tensor.long(), num_classes=self.out_dim).float()
    
    def infer_output_info(self, vshape_in: Tuple, dtype_in: torch.dtype) -> Tuple:
        return (self.out_dim,), torch.float32