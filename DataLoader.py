from dataset import myDataset
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import transforms

class DataLoader():

    @classmethod
    def GetDataLoaders(sel, path_low='data/bothDatasets/Low', path_high='data/bothDatasets/High', batch_size=64, train_size=0.9, val_size=0.05):
        transform = transforms.Compose([
            transforms.Resize([400, 400]),
            transforms.ToTensor()
        ])
        
        my_dataset = myDataset(path_low, path_high, transform)
        print(len(my_dataset))

        # Getting size of sets 
        dataset_len = len(my_dataset)
        num_train = int(np.floor(dataset_len * train_size))
        num_val = int(np.floor(dataset_len * val_size))
        num_test = dataset_len - num_train - num_val
        print(f'Num of train images: {num_train}, num of val images: {num_val}, num of test images: {num_test}')

        # Getting sets
        train_set, val_set, test_set = data.random_split(my_dataset, [num_train, num_val, num_test])

        # Getting DataLoaders
        train_dataloader = data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
        test_dataloader = data.DataLoader(test_set, batch_size = batch_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader
        
# # # Display image and label.
# train_dataloader, _, _ = DataLoader().GetDataLoaders()
# print("Hello")
# train_LOW, train_HIGH = next(iter(train_dataloader))
# print(f"train_LOW shape: {train_LOW.size()}")
# print(f"train_HIGH shape: {train_HIGH.size()}")
# imgHIGH = train_HIGH[0].squeeze()
# imgLOW = train_LOW[0].squeeze()

# imgLOW = transforms.ToPILImage(mode='RGB')(imgLOW)
# imgHIGH = transforms.ToPILImage(mode='RGB')(imgHIGH)

# imgLOW.show()
# imgHIGH.show()