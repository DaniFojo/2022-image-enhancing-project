import torch
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
import wandb
from model.model import DecomNet, RelightNet, RelightNetConvTrans, loss_decom_net, loss_relight_net
from dataloader import MyDataLoader
from logger import WandbLogger
import os
from datetime import datetime


# Constants
N_EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
	print("WARNING: running on cpu.")

logger = WandbLogger()


def train_decom(model_decom, train_loader, opt):
    losses_decom = []
    model_decom.train()
    for j, (img_low, img_high) in enumerate(train_loader):
        opt.zero_grad()
        img_low, img_high = img_low.to(device), img_high.to(device)
        r_low, i_low = model_decom(img_low)
        r_high, i_high = model_decom(img_high)
        loss = loss_decom_net(img_low, img_high, r_low, i_low, r_high, i_high)
        loss.backward()
        opt.step()
        losses_decom.append(loss.item())
    loss_mean_decomposition = np.mean(losses_decom)
    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, i_high=i_high, r_low=r_low, r_high=r_high, mode='tr', net='decom')
    logger.log_loss(loss=loss_mean_decomposition, mode='tr', net='decom')


def train_relight(model_decom, model_rel, train_loader, opt):
    losses_relight = []
    model_decom.eval()
    model_rel.train()
    for j, (img_low, img_high) in enumerate(train_loader):
        opt.zero_grad()
        img_low, img_high = img_low.to(device), img_high.to(device)
        r_low, i_low = model_decom(img_low)
        i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
        loss = loss_relight_net(img_high, r_low, i_enhanced)
        loss.backward()
        opt.step()
        losses_relight.append(loss.item())
    i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
    reconstructed = r_low * i_enhanced_3
    loss_mean_relight = np.mean(losses_relight)
    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, r_low=r_low, i_enhanced=i_enhanced, reconstructed=reconstructed, mode='tr', net='rel')
    logger.log_loss(loss=loss_mean_relight, mode='tr', net='rel')


def eval_decom(model_decom, val_loader):
    losses = []
    with torch.no_grad():
        model_decom.eval()
        for j, (img_low, img_high) in enumerate(val_loader):
            img_low, img_high = img_low.to(device), img_high.to(device)
            r_low, i_low = model_decom(img_low)
            r_high, i_high = model_decom(img_high)
            loss = loss_decom_net(img_low, img_high, r_low, i_low, r_high, i_high)
            losses.append(loss.item())
        loss_mean_decomposition = np.mean(losses)
        logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, i_high=i_high, r_low=r_low, r_high=r_high, mode='vl', net='decom')
        logger.log_loss(loss=loss_mean_decomposition, mode='vl', net='decom')
    return loss_mean_decomposition


def eval_relight(model_decom, model_rel, val_loader):
    losses = []
    with torch.no_grad():
        model_decom.eval()
        model_rel.eval()
        for img_low, img_high in val_loader:
            img_low, img_high = img_low.to(device), img_high.to(device)
            r_low, i_low = model_decom(img_low)
            i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
            loss = loss_relight_net(img_high, r_low, i_enhanced)
            losses.append(loss.item())
            i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
            reconstructed = r_low * i_enhanced_3
        loss_mean_relight = np.mean(losses)
        logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, r_low=r_low, i_enhanced=i_enhanced, reconstructed=reconstructed, mode='vl', net='rel')
        logger.log_loss(loss=loss_mean_relight, mode='vl', net='rel')  
    return loss_mean_relight


def save_model(model, optimizer, epoch, savedir):
    print(f"Saving checkpoint to {savedir}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, savedir)


def load_model(checkpoint_path, model_name):
    checkpoint = torch.load(checkpoint_path)
    if model_name == 'decom':
        model = DecomNet().to(device)
    else:
        model = RelightNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


if __name__ == "__main__":

    # Hyperparameters
    DECOM_NET_LR = 0.001
    RELIGHT_NET_LR = 0.001

    # Get DataLoaders
    train_data_loader, val_data_loader, test_data_loader \
        = MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low', 
                                          path_high='/opt/proj_img_enhance/data/train/high')

    # Load the model blocks:
    model_decomposition = DecomNet().to(device)
    model_relight = RelightNet().to(device)

    # Define optimizers:
    optimizer_decomposition = optim.Adam(model_decomposition.parameters(), DECOM_NET_LR)
    optimizer_relight = optim.Adam(model_relight.parameters(), RELIGHT_NET_LR)

    # ReduceLROnPlateau:
	# Reduce learning rate when a metric has stopped improving.
    scheduler_decomposition = optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposition, patience=5, factor=0.5)
    scheduler_relight = optim.lr_scheduler.ReduceLROnPlateau(optimizer_relight, patience=5, factor=0.5)

    # StepLR:
	# Decays the learning rate of each parameter group by gamma every step_size epochs.
    # scheduler_decom = optim.lr_scheduler.StepLR(optimizer_decomposition, step_size=1, gamma=0.95)
    # scheduler_relight = optim.lr_scheduler.StepLR(optimizer_relight, step_size=1, gamma=0.95)

    date_dir = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    path_decomposition = os.path.join('checkpoints', 'decomposition', date_dir)
    os.mkdir(path_decomposition)
    path_relight = os.path.join('checkpoints', 'relight', date_dir)
    os.mkdir(path_relight)

    for epoch in range(1, N_EPOCHS+1):
        print(f"Epoch: {epoch}")

        print("Training Decompostion")
        train_decom(model_decomposition, train_data_loader, optimizer_decomposition)

        print("Validation Decompostion")
        loss_mean_decomposition = eval_decom(model_decomposition, val_data_loader)
        scheduler_decomposition.step(loss_mean_decomposition)
        logger.log_learning_rate(optimizer_decomposition, net='decom')

        if epoch % 10 == 0:
            savedir = os.path.join(path_decomposition, f"model_decomposition_epoch_{epoch}.pt")
            save_model(model_decomposition, optimizer_decomposition, epoch, savedir)


    for epoch in range(1, N_EPOCHS+1):
        print(f"Epoch: {epoch}")

        print("Training Relight")
        train_relight(model_decomposition, model_relight, train_data_loader, optimizer_relight)

        print("Validation Relight")
        loss_mean_relight = eval_relight(model_decomposition, model_relight, val_data_loader)
        scheduler_relight.step(loss_mean_relight)
        logger.log_learning_rate(optimizer_relight, net='rel')

        if epoch % 10 == 0:
            savedir = os.path.join(path_relight, f"model_relight_epoch_{epoch}.pt")
            save_model(model_relight, optimizer_relight, epoch, savedir)



    # path_decomposition = os.path.join('checkpoints', 'decomposition', '2022_04_07_13_15_25','model_decomposition_epoch_1.pt')
    # path_relight = os.path.join('checkpoints', 'relight', '2022_04_07_13_15_25', 'model_relight_epoch_1.pt')
    # model_decomposition = load_model(path_decomposition, 'decom')
    # model_relight = load_model(path_relight, 'relight')

    # eval_decom(model_decomposition, test_data_loader)
    # eval_relight(model_decomposition, model_relight, test_data_loader)


