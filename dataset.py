import os, os.path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob

class myDataset(Dataset):

    def __init__(self, images_path_low, images_path_high, transform=None):
        super().__init__()
        self.images_path_high = images_path_high
        self.images_path_low = images_path_low
        self.transform = transform
        self.name_index_list = [os.path.basename(name) for name in glob.glob(os.path.join(self.images_path_high, "./*.png"))]

    def __len__(self):
        return len(self.name_index_list)


    def __getitem__(self, idx):
        #print(f"index = {idx}, name = {self.name_index_list[idx]}, path = {self.images_path_low}")
        path_low  =  os.path.join(self.images_path_low, self.name_index_list[idx])
        path_high =  os.path.join(self.images_path_high, self.name_index_list[idx])

        sample_low = Image.open(path_low)
        sample_high = Image.open(path_high)
        if self.transform:
            sample_low = self.transform(sample_low)
            sample_high = self.transform(sample_high)

        return sample_high, sample_low
