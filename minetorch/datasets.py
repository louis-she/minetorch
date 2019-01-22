import pandas as pd
from torch.utils.data import Dataset


class MineDataset(Dataset):

    def __init__(self, data, processor=None):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # processor will modifed data inplace, `copy()`
        # to prevent self.data being modified
        data = self.data[index].copy()
        if self.processor is not None:
            data = self.processor(data)
        return data


class CsvDataset(MineDataset):

    def __init__(self, csv_file_path, processor=None):
        data = self.generate_data(csv_file_path)
        super().__init__(data, processor)

    def generate_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        return df.values.tolist()
