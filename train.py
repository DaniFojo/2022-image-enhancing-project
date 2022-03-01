import torch
# from model.model import DecomNet, RelightNet, ImageEnhance, loss_decomNet, loss_relightNet
import model.model as retinex_model
import torch.optim as optim
from DataLoader import DataLoader
import numpy as np


# Constants
N_EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_decom, model_relight, train_loader, optimizer_decom, optimizer_relight):
	losses_decom = []
	losses_relight = []
	step = 0
	print("starting...")
	for img_low, img_normal in train_loader:
		model_decom.train()
		model_relight.eval()

		optimizer_decom.zero_grad()
		img_low, img_normal = img_low.to(device), img_normal.to(device)
		
		r_low, i_low 	= model_decom(img_low)
		r_norm, i_norm 	= model_decom(img_normal)

		loss_decom = retinex_model.loss_decomNet(img_low, img_normal, r_low, i_low, r_norm, i_norm)
		loss_decom.backward()
		optimizer_decom.step()
		losses_decom.append(loss_decom.item())


		model_decom.eval()
		model_relight.train()
		optimizer_relight.zero_grad()

		i_enhanced 		= model_relight(torch.concat((r_low,i_low), dim=1))

		loss_relight = retinex_model.loss_relightNet(img_normal, r_low, i_enhanced) # i_delta = illumination delta - output of RelightNet (enhanced illumination for the low-light image)
		loss_relight.backward()
		optimizer_relight.step()
		losses_relight.append(loss_relight.item())

		print(f"Step: {step}, loss: {loss_decom.item()}")
		step += 1

	return np.mean(losses_decom), np.mean(loss_relight)


def train_decom(model_decom, train_loader, optimizer):
	losses = []
	step = 0
	model_decom.train()
	print("starting...")
	for img_low, img_normal in train_loader:
		optimizer.zero_grad()
		img_low, img_normal = img_low.to(device), img_normal.to(device)
		
		r_low, i_low 	= model_decom(img_low)
		r_norm, i_norm 	= model_decom(img_normal)

		loss = retinex_model.loss_decomNet(img_low, img_normal, r_low, i_low, r_norm, i_norm)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		print(f"Step: {step}, loss: {loss.item()}")
		step += 1

	return np.mean(losses)



def train_relight(model_decom, model_relight, train_loader, optimizer):
	losses = []
	step = 0
	model_decom.eval()
	model_relight.train()
	for img_low, img_normal in train_loader:
		optimizer.zero_grad()
		img_low, img_normal = img_low.to(device), img_normal.to(device)
		
		r_low, i_low 	= model_decom(img_low)	# esta parte de la red ya estaría entrenada. 
												# Solo habría que hacer el forward para 
												# obtener reflectance e illumination.
		i_enhanced 		= model_relight(torch.concat((r_low,i_low), dim=1))

		loss = retinex_model.loss_relightNet(img_normal, r_low, i_enhanced) # i_delta = illumination delta - output of RelightNet (enhanced illumination for the low-light image)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		print(f"Step: {step}, loss: {loss.item()}")
		step += 1

	return np.mean(losses)




def eval_decom(model_decom, val_loader):
	losses = []

	with torch.no_grad():
		model_decom.eval()
		for img_low, img_normal in val_loader:
			img_low, img_normal = img_low.to(device), img_normal.to(device)

			r_low, i_low 	= model_decom(img_low)
			r_norm, i_norm 	= model_decom(img_normal)

			loss = retinex_model.loss_decomNet(img_low, img_normal, r_low, i_low, r_norm, i_norm)
			losses.append(loss.item())
			
	return np.mean(losses)



def eval_relight(model_decom, model_relight, val_loader):
	losses = []

	with torch.no_grad():
		model_decom.eval()
		model_relight.eval()
		for img_low, img_normal in val_loader:
			img_low, img_normal = img_low.to(device), img_normal.to(device)
			
			r_low, i_low 	= model_decom(img_low)
			i_enhanced 		= model_relight(torch.concat(r_low,i_low))

			loss = retinex_model.loss_relightNet(img_normal, r_low, i_enhanced) # i_delta = illumination delta - output of RelightNet (enhanced illumination for the low-light image)
			losses.append(loss.item())

	return np.mean(losses)




if __name__ == "__main__":
	# * Aqui o en el main.py. Para entrenar el modelo.

	# Hyperparameters
	DECOM_NET_LR 	= 0.001
	RELIGHT_NET_LR 	= 0.001

	#################

	# Get DataLoaders
	train_loader, val_loader, test_loader = DataLoader.GetDataLoaders()

	# Load the model blocks:
	model_decom 	= retinex_model.DecomNet()
	model_relight 	= retinex_model.RelightNet()
  
	# Define optimizers:
	optimizer_decom 	= optim.Adam(model_decom.parameters(), DECOM_NET_LR)
	optimizer_relight 	= optim.Adam(model_relight.parameters(), RELIGHT_NET_LR)


	# TRAIN:
	print("Starting train")
	for epoch in range(N_EPOCHS):
		print(f"Epoch: {epoch}")
		# print("Training Decom")
		# train_decom_loss 	= train_decom(model_decom, train_loader, optimizer_decom)
		# print("Training Relight")
		# train_relight_loss 	= train_relight(model_decom, model_relight, train_loader, optimizer_relight)

		train_decom_loss, train_relight_loss = train(model_decom, model_relight, train_loader, optimizer_decom, optimizer_relight)


		val_decom_loss 		= eval_decom(model_decom, val_loader)
		val_relight_loss 	= eval_relight(model_decom, model_relight, val_loader)


	# TEST

	# SAVE
	pass