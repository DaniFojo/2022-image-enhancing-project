import argparse
import torch
from model.model import DecomNet, RelightNet, ImageEnhance


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument("--n_samples", help="amount of samples to train with", type=int, default=1000)
# args = parser.parse_args()


# # parameters
# num_epochs = 20
# lr = 0.0002
# betas = (0.5, 0.999)
# noise_size = 100
# batch_size = 128
# num_val_samples = 25
# num_classes = 10
# num_input_channels = 1


# decom = DecomNet().to(device)
# optimizer_d = torch.optim.Adam(decom.parameters(), lr=lr, betas=betas)
# relight = RelightNet().to(device)
# optimizer_r = torch.optim.Adam(relight.parameters(), lr=lr, betas=betas)

x = torch.randn(1,3,50,50)
print(x.shape)

model = ImageEnhance()
forw = model(x)
print(forw)
