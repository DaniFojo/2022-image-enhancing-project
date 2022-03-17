import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import MyDataset


class MyDataLoader():

    def get_data_loaders(self, path_low='data/low', path_high='data/high',
                         batch_size=16, train_size=0.9, val_size=0.05):
        print("get_data_loaders: path low: ", path_low, " - path high: ", path_high)
        transform = transforms.Compose(
            [transforms.Resize([300, 300]), transforms.ToTensor()])
        my_dataset = MyDataset(path_low, path_high, transform)

        # Getting size of sets
        dataset_len = len(my_dataset)
        assert dataset_len > 0
        num_train = int(np.floor(dataset_len * train_size))
        num_val = int(np.floor(dataset_len * val_size))
        num_test = dataset_len - num_train - num_val
        print(
            f'Num of train images: {num_train}, num of val images: {num_val}, \
              num of test images: {num_test}')

        # Getting sets
        train_set, val_set, test_set = random_split(
            my_dataset, [num_train, num_val, num_test])

        # Getting DataLoaders
        train_dataloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader
