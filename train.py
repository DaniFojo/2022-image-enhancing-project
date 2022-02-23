import torch
import time
from torch.utils.data import DataLoader
import model.model as m


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataloader):
    model.train()
    pass

def eval(dataloader):
    model.eval()
    pass

def test(dataloader):
    model.eval()
    pass

if __name__ == "__main__":
    # * Aqui o en el main.py. Para entrenar el modelo.

    # Hyperparameters

    #################
    
    # Load the dataset
    # Load the model

    # TRAIN

    # SAVE
    pass