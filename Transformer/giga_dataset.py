from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import pickle

class GigaDataset(Dataset):
    def __init__(self, processed_dir, transform = None) -> None:
        self.processed_dir = processed_dir
        self.processed_file_names = [name for name in os.listdir(processed_dir)]
        super(GigaDataset, self).__init__(processed_dir=processed_dir, transform=transform)
    
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int):
        file = self.processed_dir + self.processed_file_names[idx]
        with open(file, 'rb') as handle:
            return HeteroData(pickle.load(handle))







                   
                   
