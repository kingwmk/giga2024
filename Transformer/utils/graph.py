from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import coalesce
from torch_geometric.utils import degree

def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)
