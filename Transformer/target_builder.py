import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle

class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        origin_last = data['agent']['position'][:, self.num_historical_steps - 2]
        motion_vector = origin - origin_last
        theta = torch.atan2(motion_vector[:, 1], motion_vector[:, 0])
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 2)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)

        return data
