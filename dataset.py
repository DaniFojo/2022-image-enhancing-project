from torch.utils.data import Dataset

# Dataset functions


class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        # self.labels
        # self.images
        # self.transform
        pass


    def __len__(self):
        # len(...)
        pass


    def __getitem__(self, idx):
        # return self.transfrom(image), id
        pass