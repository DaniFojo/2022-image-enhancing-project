import os
from datetime import datetime
import numpy as np
import torch
from torch import optim
from model.model import DecomNet, RelightNet, RelightNetConvTrans
from model.model import loss_decom_net, loss_relight_net
from dataloader import MyDataLoader
from logger import WandbLogger
from utils import save_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("WARNING: running on cpu.")

logger = WandbLogger()


def train_decom(model_decom, train_loader, opt, epoch):
    losses_decom = []
    model_decom.train()
    img_low, img_high = None, None
    for _, (img_low, img_high) in enumerate(train_loader):
        opt.zero_grad()
        img_low, img_high = img_low.to(device), img_high.to(device)
        r_low, i_low = model_decom(img_low)
        r_high, i_high = model_decom(img_high)
        loss = loss_decom_net(img_low, img_high, r_low, i_low, r_high, i_high)
        loss.backward()
        opt.step()
        losses_decom.append(loss.item())
    loss_mean_decomposition = np.mean(losses_decom)
    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low,
                           i_high=i_high, r_low=r_low, r_high=r_high,
                           mode='tr', net='decom', step=epoch)
    logger.log_loss(loss=loss_mean_decomposition, mode='tr', net='decom',
                    step=epoch)


def train_relight(model_decom, model_rel, train_loader, opt, epoch):
    losses_relight = []
    model_decom.eval()
    model_rel.train()
    img_low, img_high = None, None
    for _, (img_low, img_high) in enumerate(train_loader):
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
    logger.log_images_grid(img_low=img_low, img_high=img_high,
                           i_low=i_low, r_low=r_low,
                           i_enhanced=i_enhanced,
                           reconstructed=reconstructed,
                           mode='tr', net='rel', step=epoch)
    logger.log_loss(loss=loss_mean_relight, mode='tr', net='rel',
                    step=epoch)


def eval_decom(model_decom, val_loader, epoch):
    losses = []
    with torch.no_grad():
        model_decom.eval()
        img_low, img_high = None, None
        for _, (img_low, img_high) in enumerate(val_loader):
            img_low, img_high = img_low.to(device), img_high.to(device)
            r_low, i_low = model_decom(img_low)
            r_high, i_high = model_decom(img_high)
            loss = loss_decom_net(img_low, img_high, r_low, i_low,
                                  r_high, i_high)
            losses.append(loss.item())
        loss_mean_decomposition = np.mean(losses)
        logger.log_images_grid(img_low=img_low, img_high=img_high,
                               i_low=i_low, i_high=i_high,
                               r_low=r_low, r_high=r_high,
                               mode='vl', net='decom', step=epoch)
        logger.log_loss(loss=loss_mean_decomposition, mode='vl', net='decom',
                        step=epoch)
    return loss_mean_decomposition


def eval_relight(model_decom, model_rel, val_loader, epoch):
    losses = []
    with torch.no_grad():
        model_decom.eval()
        model_rel.eval()
        img_low, img_high = None, None
        for img_low, img_high in val_loader:
            img_low, img_high = img_low.to(device), img_high.to(device)
            r_low, i_low = model_decom(img_low)
            i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
            loss = loss_relight_net(img_high, r_low, i_enhanced)
            losses.append(loss.item())
            i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced),
                                        dim=1)
            reconstructed = r_low * i_enhanced_3
        loss_mean_relight = np.mean(losses)
        logger.log_images_grid(img_low=img_low, img_high=img_high,
                               i_low=i_low, r_low=r_low,
                               i_enhanced=i_enhanced,
                               reconstructed=reconstructed,
                               mode='vl', net='rel', step=epoch)
        logger.log_loss(loss=loss_mean_relight, mode='vl', net='rel',
                        step=epoch)
    return loss_mean_relight


def training_split(n_epochs, decom_lr, relight_lr, s_epochs, transposed):
    train_data_loader, val_data_loader, _ \
        = MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low',
                                          path_high='/opt/proj_img_enhance/data/train/high')

    model_decomposition = DecomNet().to(device)
    if transposed:
        model_relight = RelightNetConvTrans().to(device)
    else:
        model_relight = RelightNet().to(device)

    optimizer_decomposition = optim.Adam(model_decomposition.parameters(),
                                         decom_lr)
    optimizer_relight = optim.Adam(model_relight.parameters(), relight_lr)

    scheduler_decomposition = optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposition,
                                                                   patience=5,
                                                                   factor=0.5)
    scheduler_relight = optim.lr_scheduler.ReduceLROnPlateau(optimizer_relight,
                                                             patience=5,
                                                             factor=0.5)

    date_dir = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    path_decomposition = os.path.join('checkpoints', 'split', 'decomposition',
                                      date_dir)
    os.mkdir(path_decomposition)
    path_relight = os.path.join('checkpoints', 'split', 'relight', date_dir)
    os.mkdir(path_relight)

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")

        print("Training Decompostion")
        train_decom(model_decomposition, train_data_loader,
                    optimizer_decomposition, epoch)

        print("Validation Decompostion")
        loss_mean_decomposition = eval_decom(model_decomposition,
                                             val_data_loader, epoch)
        scheduler_decomposition.step(loss_mean_decomposition)
        logger.log_learning_rate(optimizer_decomposition, net='decom',
                                 step=epoch)

        if epoch % s_epochs == 0:
            savedir = os.path.join(path_decomposition,
                                   f"model_decomposition_epoch_{epoch}.pt")
            save_model(model_decomposition, optimizer_decomposition, epoch,
                       savedir)

    for epoch in range(n_epochs + 1, 2 * n_epochs + 1):
        print(f"Epoch: {epoch}")

        print("Training Relight")
        train_relight(model_decomposition, model_relight, train_data_loader,
                      optimizer_relight, epoch)

        print("Validation Relight")
        loss_mean_relight = eval_relight(model_decomposition, model_relight,
                                         val_data_loader, epoch)
        scheduler_relight.step(loss_mean_relight)
        logger.log_learning_rate(optimizer_relight, net='rel', step=epoch)

        if epoch % s_epochs == 0:
            savedir = os.path.join(path_relight,
                                   f"model_relight_epoch_{epoch}.pt")
            save_model(model_relight, optimizer_relight, epoch, savedir)
