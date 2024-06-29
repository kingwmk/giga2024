from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import pickle

class GigaDataset(Dataset):
    def __init__(self, processed_dir, transform = None) -> None:
        self._processed_dir = processed_dir
        self._processed_file_names = [name for name in os.listdir(processed_dir)]
        super(GigaDataset, self).__init__(processed_dir=processed_dir, transform=transform)
        
    @property
    def processed_dir(self):
        return self._processed_dir

    @processed_dir.setter
    def processed_dir(self, value):
        self._processed_dir = value    
        
    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int):
        file = self._processed_dir + self._processed_file_names[idx]
        with open(file, 'rb') as handle:
            data = dict()
            data['agent'] = pickle.load(handle)
            return HeteroData(data)







                   
                   
