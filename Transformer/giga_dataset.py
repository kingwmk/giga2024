from torch_geometric.data import Dataset

class ArgoverseV2Dataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 2,
                 num_historical_steps: int = 60,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 num_recurrent_steps = 3,
                 occupancy_threshold: float = 2) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        self.split = split
        self._raw_dir = os.path.join(root, split)
        self._raw_file_names = [name for name in os.listdir(self._raw_dir)]
        



                   
                   
