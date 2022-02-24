import os, os.path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class myDataset(Dataset):

    def __init__(self, images_path_low, images_path_high, transform=None):
        super().__init__()
        # lista con indices
        self.images_path_high = images_path_high
        self.images_path_low = images_path_low
        self.transform = transform
        self.data_length = len([name for name in os.listdir(self.images_path_high)]) # Use Glob to filter by png.

    def __len__(self):
        # len => Number of files inside the directory images_path_high.
        return self.data_length


    def __getitem__(self, idx):
        path_low =  os.path.join(self.images_path_low, f"{idx}.png")
        path_high =  os.path.join(self.images_path_high, f"{idx}.png")

        # Fix number of files to solve this. Provisional fix
        while not os.path.isfile(path_low) and idx < self.data_lenght:
            idx += 1
            path_low =  os.path.join(self.images_path_low, f"{idx}.png")
            path_high =  os.path.join(self.images_path_high, f"{idx}.png")

        sample_low = Image.open(path_low)
        sample_high = Image.open(path_high)
        if self.transform:
            sample_low = self.transform(sample_low)
            sample_high = self.transform(sample_high)
        return sample_high, sample_low
