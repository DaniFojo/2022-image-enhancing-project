import torch
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
import wandb
from model.model import DecomNet, RelightNet, loss_decom_net, loss_relight_net
from dataloader import MyDataLoader
from logger import WandbLogger
import os
from datetime import datetime


# Constants
N_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = WandbLogger()


def forward_losses(model_decom, model_rel, img_low, img_high):
    img_low, img_high = img_low.to(device), img_high.to(device)
    r_low, i_low = model_decom(img_low)
    r_high, i_high = model_decom(img_high)
    r_low, i_low = r_low.to(device), i_low.to(device)
    r_high, i_high = r_high.to(device), i_high.to(device)
    loss_decom = loss_decom_net(img_low, img_high, r_low, i_low, r_high, i_high)
    i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
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
        optimizer.zero_grad()
        i_low, i_high, r_low, r_high, i_enhanced, reconstructed, loss_decom, loss_relight, loss = forward_losses(model_decom, model_rel, img_low, img_high)
        losses_decom.append(loss_decom.item())
        losses_relight.append(loss_relight.item())
        losses_total.append(loss.item())
        if j % 5 == 0:
            logger.log_images_grid(img_low, img_high, i_low, i_high, r_low, r_high, i_enhanced, reconstructed, mode='train')
            logger.log_training(loss, loss_decom, loss_relight)
        loss.backward()
        opt.step()
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
            logger.log_images_grid(img_low, img_high, i_low, i_high, r_low, r_high, i_enhanced, reconstructed, mode='validation')
            logger.log_eval(loss, loss_decom, loss_relight)
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
        if j % 5 == 0:
            logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low, i_high=i_high, r_low=r_low, r_high=r_high, mode='train')
            logger.log_training(loss_decom=loss)
    return np.mean(losses_decom)


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
        if j % 5 == 0:
            i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
            reconstructed = r_low * i_enhanced_3
            logger.log_images_grid(i_enhanced=i_enhanced, reconstructed=reconstructed, mode='train')
            logger.log_training(loss_relight=loss)
    return np.mean(losses_relight)


def eval_decom(model_decom, val_loader):
    losses = []

    with torch.no_grad():
        model_decom.eval()
        for img_low, img_high in val_loader:
            img_low, img_high = img_low.to(device), img_high.to(device)
            r_low, i_low = model_decom(img_low)
            r_high, i_high = model_decom(img_high)
            loss = loss_decom_net(img_low, img_high, r_low, i_low, 
                                  r_high, i_high)
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
            loss = loss_relight_net(img_high, r_low, i_enhanced)
            # i_delta = illumination delta - output of RelightNet
            # (enhanced illumination for the low-light image)
            losses.append(loss.item())
    return np.mean(losses)


def save_model(model_decom, model_relight, optimizer, epoch, savedir):
    # Save the artifacts of the training
    print(f"Saving checkpoint to {savedir}...")
    # We can save everything we will need later in the checkpoint.
    checkpoint = {
        "model_decom_state_dict": model_decom.cpu().state_dict(),
        "model_relight_state_dict": model_relight.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, savedir)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model_decomposition = DecomNet().to(device)
    model_relight = RelightNet().to(device)

    model_decomposition.load_state_dict(checkpoint["model_decom_state_dict"])
    model_relight.load_state_dict(checkpoint["model_relight_state_dict"])

    return model_decomposition, model_relight


if __name__ == "__main__":

    # Hyperparameters
    DECOM_NET_LR = 0.001
    RELIGHT_NET_LR = 0.001

    wandb.login(relogin=True)
    wandb.init(project="retinex")  # añadir hyperparameters en el init de wandb

    # Get DataLoaders
    train_data_loader, val_data_loader, test_data_loader \
        = MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low', 
                                          path_high='/opt/proj_img_enhance/data/train/high')

    # Load the model blocks:
    model_decomposition = DecomNet().to(device)
    model_relight = RelightNet().to(device)

    # Define optimizers:
    optimizer_decom = optim.Adam(model_decomposition.parameters(),
                                 DECOM_NET_LR)
    optimizer_relight = optim.Adam(model_relight.parameters(),
                                   RELIGHT_NET_LR)

    optimizer = optim.Adam([{"params": model_decomposition.parameters(),
                             "lr": DECOM_NET_LR},
                            {"params": model_relight.parameters(),
                             "lr": RELIGHT_NET_LR}])

    epoch_train_decom_losses = []
    epoch_train_relight_losses = []
    epoch_train_total_losses = []
    epoch_val_decom_losses = []
    epoch_val_relight_losses = []
    epoch_val_total_losses = []    
    for epoch in range(N_EPOCHS):
        print(f"Epoch: {epoch}")
        print("Training")
        train_decom_loss, train_relight_loss, train_total_loss = train(model_decomposition, model_relight, train_data_loader, optimizer)
        epoch_train_decom_losses.append(train_decom_loss)
        epoch_train_relight_losses.append(train_relight_loss)
        epoch_train_total_losses.append(train_total_loss)

        print("Validation")
        val_decom_loss, val_relight_loss, val_total_loss = eval(model_decomposition, model_relight, val_data_loader)
        epoch_val_decom_losses.append(val_decom_loss)
        epoch_val_relight_losses.append(val_relight_loss)
        epoch_val_total_losses.append(val_total_loss)

        savedir = os.path.join('checkpoints/', datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        save_model(model_decomposition, model_relight, optimizer, epoch, savedir)

