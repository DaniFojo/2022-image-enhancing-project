import torch
import time

from model.model import DecomNet, RelightNet, ImageEnhance, loss_decomNet, loss_relightNet
import torch.optim as optim

from dataset import Dataset


# Constants
N_EPOCHS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_decom(train_loader, optimizer):
	losses = []

	DecomNet.train()
	for img_low, img_normal in train_loader:
		optimizer.zero_grad()
		
		r_low, i_low 	= DecomNet(img_low)
		r_norm, i_norm 	= DecomNet(img_normal)
		loss = loss_decomNet(img_low, img_normal, r_low, i_low, r_norm, i_norm)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())



def train_relight(train_loader, optimizer):
	losses = []

	DecomNet.train()
	for img_low, img_normal in train_loader:
		optimizer.zero_grad()
		
		r_low, i_low 	= DecomNet(img_low)
		r_norm, i_norm 	= DecomNet(img_normal) 	# no necesario para evaluar la loss?
		i_enhanced 	= RelightNet(concat(r_low,i_low))

		loss = loss_relightNet(img_normal, r_low, i_enhanced) # i_delta = illumination delta - output of RelightNet (enhanced illumination for the low-light image)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())



if __name__ == "__main__":
	# * Aqui o en el main.py. Para entrenar el modelo.

	# Hyperparameters

	#################

	# Get DataLoaders
	train_loader, val_loader, test_loader = Dataset.getDataLoaders()

	# Load the model
	my_model = ImageEnhance()
  
	# Define optimizers:
	optimizer_decom 	= optim.Adam(...)  # fuera del for de epoch. Aplicar param groups, 1 solo optimizer.
	optimizer_relight 	= optim.Adam(...)


	# TRAIN:
	for epoch in range(N_EPOCHS):
		train_decom_loss = train_decom(train_loader, optimizer_decom)
		train_relight_loss = train_relight(train_loader, optimizer_relight)
		
		val_decom_loss = eval_decom(val_loader)
		val_relight_loss = eval_relight(val_loader)


	# TEST

	# SAVE
	pass