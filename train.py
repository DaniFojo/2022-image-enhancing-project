import torch
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
import wandb
import model.model as retinex_model
from dataloader import MyDataLoader



# Constants
N_EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_decom, model_rel, train_loader, opt):
	losses_decom = []
	losses_relight = []
	loss_total = []

	print("starting...")
	model_decom.train()
	model_relight.train()
	for img_low, img_high in train_loader:
		grid_low = make_grid(img_low)
		grid_high = make_grid(img_high)
		wdb_low = wandb.Image(grid_low, "input_image_low")
		wdb_high = wandb.Image(grid_high, "input_image_high")

		optimizer.zero_grad()
		img_low, img_high = img_low.to(device), img_high.to(device)
		r_low, i_low = model_decom(img_low)
		r_norm, i_norm = model_decom(img_high)

		grid_ilow = make_grid(i_low)
		grid_ihigh = make_grid(i_norm)
		wdb_ilow = wandb.Image(grid_ilow, "ilow")
		wdb_ihigh = wandb.Image(grid_ihigh, "ihigh")

		grid_rlow = make_grid(r_low)
		grid_rhigh = make_grid(r_norm)
		wdb_rlow = wandb.Image(grid_rlow, "rlow")
		wdb_rhigh = wandb.Image(grid_rhigh, "rhigh")

		r_low, i_low = r_low.to(device), i_low.to(device)
		r_norm, i_norm = r_norm.to(device), i_norm.to(device)

		loss_decom = retinex_model.loss_decom_net(img_low, img_high, r_low, i_low, r_norm, i_norm)
		losses_decom.append(loss_decom.item())

		i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
		grid_ienhanced = make_grid(i_enhanced)
		wdb_ienhanced = wandb.Image(grid_ienhanced, "ienhanced")

		loss_relight = retinex_model.loss_relight_net(img_high, r_low, i_enhanced)
		losses_relight.append(loss_relight.item())

		loss = loss_decom + loss_relight
		loss_total.append(loss.item())
		loss.backward()
		opt.step()

		wandb.log({"loss_total": loss_total, "loss_decom": loss_decom, "loss_relight": loss_relight,
				   "image_low": wdb_low, "image_high": wdb_high, "ilow": wdb_ilow,
				   "ihigh": wdb_ihigh, "rlow": wdb_rlow, "rhigh": wdb_rhigh, "ienhanced": wdb_ienhanced})

	return np.mean(losses_decom), np.mean(losses_relight)


def train_decom(model_decom, train_loader, opt):
	losses = []
	step = 0
	model_decom.train()
	print("starting...")
	for img_low, img_high in train_loader:
		opt.zero_grad()
		img_low, img_high = img_low.to(device), img_high.to(device)
		r_low, i_low = model_decom(img_low)
		r_norm, i_norm = model_decom(img_high)
		loss = retinex_model.loss_decom_net(img_low, img_high, r_low, i_low, r_norm, i_norm)
		loss.backward()
		opt.step()
		losses.append(loss.item())
		print(f"Step: {step}, loss: {loss.item()}")
		step += 1

	return np.mean(losses)


def train_relight(model_decom, model_rel, train_loader, opt):
	losses = []
	step = 0
	model_decom.eval()
	model_rel.train()
	for img_low, img_high in train_loader:
		opt.zero_grad()
		img_low, img_high = img_low.to(device), img_high.to(device)
		r_low, i_low = model_decom(img_low)
		# esta parte de la red ya estaría entrenada.
		# Solo habría que hacer el forward para
		# obtener reflectance e illumination.
		i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))

		loss = retinex_model.loss_relight_net(img_high, r_low, i_enhanced)
		# i_delta = illumination delta - output of RelightNet
		# (enhanced illumination for the low-light image)
		loss.backward()
		opt.step()
		losses.append(loss.item())
		print(f"Step: {step}, loss: {loss.item()}")
		step += 1

	return np.mean(losses)


def eval_decom(model_decom, val_loader):
	losses = []

	with torch.no_grad():
		model_decom.eval()
		for img_low, img_high in val_loader:
			img_low, img_high = img_low.to(device), img_high.to(device)
			r_low, i_low = model_decom(img_low)
			r_norm, i_norm = model_decom(img_high)
			loss = retinex_model.loss_decom_net(img_low, img_high, r_low, i_low, r_norm, i_norm)
			losses.append(loss.item())
	return np.mean(losses)


def eval_relight(model_decom, model_rel, val_loader):
	losses = []

	with torch.no_grad():
		model_decom.eval()
		model_rel.eval()
		for img_low, img_high in val_loader:
			img_low, img_high = img_low.to(device), img_high.to(device)
			r_low, i_low = model_decom(img_low)
			i_enhanced = model_rel(torch.concat(r_low, i_low))
			loss = retinex_model.loss_relight_net(img_high, r_low, i_enhanced)
			# i_delta = illumination delta - output of RelightNet
			# (enhanced illumination for the low-light image)
			losses.append(loss.item())
	return np.mean(losses)


if __name__ == "__main__":

	# Hyperparameters
	DECOM_NET_LR = 0.001
	RELIGHT_NET_LR = 0.001

	# wandb.login(relogin=True)
	wandb.init(project="retinex")  # añadir hyperparameters en el init de wandb

	# Get DataLoaders
	train_data_loader, val_data_loader, test_data_loader \
		= MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low',  
											path_high='/opt/proj_img_enhance/data/train/normal')

	# Load the model blocks:
	model_decomposition = retinex_model.DecomNet().to(device)
	model_relight = retinex_model.RelightNet().to(device)

	# Define optimizers:
	optimizer_decom = optim.Adam(model_decomposition.parameters(), DECOM_NET_LR)
	optimizer_relight = optim.Adam(model_relight.parameters(), RELIGHT_NET_LR)

	optimizer = optim.Adam([{"params": model_decomposition.parameters(), "lr": DECOM_NET_LR},
							{"params": model_relight.parameters(), "lr": RELIGHT_NET_LR}])

	print("Starting train")
	for epoch in range(N_EPOCHS):
		print(f"Epoch: {epoch}")
		
		# Train network end-to-end:
		train_decom_loss, train_relight_loss = train(model_decomposition, model_relight,
													 train_data_loader, optimizer)

		# Train only decomposition network:
		# train_decom(model_decomposition, train_data_loader, optimizer)      

		# val_decom_loss = eval_decom(model_decomposition, val_data_loader)
		# val_relight_loss = eval_relight(model_decomposition, model_relight, val_data_loader)
