import optparse
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
N_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
	print("WARNING: running on cpu.")

logger = WandbLogger()


def forward_losses(model_decom, model_rel, img_low, img_high):
    img_low, img_high = img_low.to(device), img_high.to(device)
    r_low, i_low = model_decom(img_low)
    r_high, i_high = model_decom(img_high)
    r_low, i_low = r_low.to(device), i_low.to(device)
    r_high, i_high = r_high.to(device), i_high.to(device)
    loss_decom = loss_decom_net(img_low, img_high, r_low, i_low, r_high, i_high)

    # Tryin with only ones
    i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
    i_enhanced = torch.ones(i_enhanced.shape).to(device)
    
    loss_relight = loss_relight_net(img_high, r_low, i_enhanced)
    loss = loss_decom + loss_relight
    i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
    reconstructed = r_low * i_enhanced_3
    return i_low, i_high, r_low, r_high, i_enhanced, reconstructed, loss_decom, loss_relight, loss


def train(model_decom, model_rel, train_loader, opt):
    losses_decom = []
    losses_relight = []
    losses_total = []
    model_decom.train()
    model_relight.train()
    for j, (img_low, img_high) in enumerate(train_loader):
        opt.zero_grad()
        i_low, i_high, r_low, r_high, i_enhanced, reconstructed, loss_decom, loss_relight, loss = forward_losses(model_decom, model_rel, img_low, img_high)
        losses_decom.append(loss_decom.item())
        losses_relight.append(loss_relight.item())
        losses_total.append(loss.item())
        loss.backward()
        opt.step()

    loss_mean_decomposition = np.mean(losses_decom)
    loss_mean_relight = np.mean(losses_relight)

    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, i_high=i_high, r_high=r_high, r_low=r_low, i_enhanced=i_enhanced, reconstructed=reconstructed, mode='tr', net='rel')
    logger.log_loss(loss=loss_mean_decomposition, mode='tr', net='decom')
    logger.log_loss(loss=loss_mean_relight, mode='tr', net='rel')

    return np.mean(losses_decom), np.mean(losses_relight), np.mean(losses_total)


def eval(model_decom, model_rel, val_loader):    
    losses_decom = []
    losses_relight = []
    losses_total = []
    with torch.no_grad():
        model_decom.eval()
        model_relight.eval()
        for j, (img_low, img_high) in enumerate(val_loader):
            i_low, i_high, r_low, r_high, i_enhanced, reconstructed, loss_decom, loss_relight, loss = forward_losses(model_decom, model_rel, img_low, img_high)
            losses_decom.append(loss_decom.item())
            losses_relight.append(loss_relight.item())
            losses_total.append(loss.item())

    loss_mean_decomposition = np.mean(losses_decom)
    loss_mean_relight = np.mean(losses_relight)

    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, i_high=i_high, r_high=r_high, r_low=r_low, i_enhanced=i_enhanced, reconstructed=reconstructed, mode='vl', net='rel')
    logger.log_loss(loss=loss_mean_decomposition, mode='vl', net='decom')
    logger.log_loss(loss=loss_mean_relight, mode='vl', net='rel')

    return np.mean(losses_decom), np.mean(losses_relight), np.mean(losses_total)



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

    TRAIN_MODE = 'JOIN'

    # Get DataLoaders
    train_data_loader, val_data_loader, test_data_loader \
        = MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low', 
                                          path_high='/opt/proj_img_enhance/data/train/high')

    # Load the model blocks:
    model_decomposition = DecomNet().to(device)
    model_relight = RelightNet().to(device)

    if TRAIN_MODE == 'JOIN':
        optimizer = optim.Adam([{"params": model_decomposition.parameters(),
                             "lr": DECOM_NET_LR},
                            {"params": model_relight.parameters(),
                             "lr": RELIGHT_NET_LR}])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        date_dir = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')


    else:

    # Define optimizers:
    #optimizer_decomposition = optim.Adam(model_decomposition.parameters(), DECOM_NET_LR)
    #optimizer_relight = optim.Adam(model_relight.parameters(), RELIGHT_NET_LR)

    # ReduceLROnPlateau:
	# Reduce learning rate when a metric has stopped improving.
    #scheduler_decomposition = optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposition, patience=5, factor=0.5)
    #scheduler_relight = optim.lr_scheduler.ReduceLROnPlateau(optimizer_relight, patience=5, factor=0.5)

    # StepLR:
	# Decays the learning rate of each parameter group by gamma every step_size epochs.
    # scheduler_decom = optim.lr_scheduler.StepLR(optimizer_decomposition, step_size=1, gamma=0.95)
    # scheduler_relight = optim.lr_scheduler.StepLR(optimizer_relight, step_size=1, gamma=0.95)

    date_dir = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    #path_decomposition = os.path.join('checkpoints', 'decomposition', date_dir)
    #os.mkdir(path_decomposition)
    #path_relight = os.path.join('checkpoints', 'relight', date_dir)
    #os.mkdir(path_relight)
    path_both = os.path.join('checkpoints', 'both', date_dir)
    os.mkdir(path_both)

    for epoch in range(1, N_EPOCHS+1):
        print(f"Epoch: {epoch}")

        print("Training")
        train_decom_loss, train_relight_loss, train_total_loss = train(model_decomposition, model_relight, train_data_loader, optimizer)
        print("Validation")
        val_decom_loss, val_relight_loss, val_total_loss = eval(model_decomposition, model_relight, val_data_loader)
        scheduler.step(val_total_loss)
        logger.log_learning_rate(optimizer, net='decom')

        if epoch % 10 == 0:
            savedir = os.path.join(path_both, f"model_decomposition_epoch_{epoch}.pt")
            save_model(model_decomposition, optimizer, epoch, savedir)

            savedir = os.path.join(path_both, f"model_relight_epoch_{epoch}.pt")
            save_model(model_relight, optimizer, epoch, savedir)


    # path_decomposition = os.path.join('checkpoints', 'decomposition', '2022_04_07_13_15_25','model_decomposition_epoch_1.pt')
    # path_relight = os.path.join('checkpoints', 'relight', '2022_04_07_13_15_25', 'model_relight_epoch_1.pt')
    # model_decomposition = load_model(path_decomposition, 'decom')
    # model_relight = load_model(path_relight, 'relight')

    # eval_decom(model_decomposition, test_data_loader)
    # eval_relight(model_decomposition, model_relight, test_data_loader)


