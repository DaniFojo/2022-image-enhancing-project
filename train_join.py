import os
from datetime import datetime
import numpy as np
import torch
from torch import optim
from model.model import DecomNet, RelightNet
from model.model import loss_decom_net, loss_relight_net
from dataloader import MyDataLoader
from logger import WandbLogger
from utils import save_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("WARNING: running on cpu.")

logger = WandbLogger()


def forward_losses(model_decom, model_rel, img_low, img_high,
                   ignore_ienhanace=False):
    img_low, img_high = img_low.to(device), img_high.to(device)
    r_low, i_low = model_decom(img_low)
    r_high, i_high = model_decom(img_high)
    r_low, i_low = r_low.to(device), i_low.to(device)
    r_high, i_high = r_high.to(device), i_high.to(device)
    loss_decom = loss_decom_net(img_low, img_high, r_low,
                                i_low, r_high, i_high)
    i_enhanced = model_rel(torch.concat((r_low, i_low), dim=1))
    if ignore_ienhanace:
        i_enhanced = torch.ones(i_enhanced.shape).to(device)
    loss_relight = loss_relight_net(img_high, r_low, i_enhanced)
    loss = loss_decom + loss_relight
    i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced),
                                dim=1)
    reconstructed = r_low * i_enhanced_3
    return i_low, i_high, r_low, r_high, i_enhanced, reconstructed, \
        loss_decom, loss_relight, loss


def train(model_decom, model_relight, train_loader, opt, epoch,
          ignore_ienhanace=False):
    losses_decom = []
    losses_relight = []
    losses_total = []
    model_decom.train()
    model_relight.train()
    img_low, img_high = None, None
    for _, (img_low, img_high) in enumerate(train_loader):
        opt.zero_grad()
        i_low, i_high, r_low, r_high, i_enhanced, reconstructed, \
            loss_decom, loss_relight, loss = forward_losses(model_decom,
                                                            model_relight,
                                                            img_low,
                                                            img_high,
                                                            ignore_ienhanace)
        losses_decom.append(loss_decom.item())
        losses_relight.append(loss_relight.item())
        losses_total.append(loss.item())
        loss.backward()
        opt.step()
    loss_mean_decomposition = np.mean(losses_decom)
    loss_mean_relight = np.mean(losses_relight)
    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low,
                           i_high=i_high, r_high=r_high, r_low=r_low,
                           i_enhanced=i_enhanced, reconstructed=reconstructed,
                           mode='tr', net='join', step=epoch)
    logger.log_loss(loss=loss_mean_decomposition, mode='tr', net='decom',
                    step=epoch)
    logger.log_loss(loss=loss_mean_relight, mode='tr', net='rel', step=epoch)


def evaluation(model_decom, model_relight, val_loader, epoch,
               ignore_ienhanace=False):
    losses_decom = []
    losses_relight = []
    losses_total = []
    with torch.no_grad():
        model_decom.eval()
        model_relight.eval()
        img_low, img_high = None, None
        for _, (img_low, img_high) in enumerate(val_loader):
            i_low, i_high, r_low, r_high, i_enhanced, reconstructed, \
                loss_decom, loss_relight, loss = forward_losses(model_decom,
                                                                model_relight,
                                                                img_low,
                                                                img_high,
                                                                ignore_ienhanace)
            losses_decom.append(loss_decom.item())
            losses_relight.append(loss_relight.item())
            losses_total.append(loss.item())
    loss_mean_decomposition = np.mean(losses_decom)
    loss_mean_relight = np.mean(losses_relight)
    logger.log_images_grid(img_low=img_low, img_high=img_high, i_low=i_low,
                           i_high=i_high, r_high=r_high, r_low=r_low,
                           i_enhanced=i_enhanced, reconstructed=reconstructed,
                           mode='vl', net='join', step=epoch)
    logger.log_loss(loss=loss_mean_decomposition, mode='vl', net='decom',
                    step=epoch)
    logger.log_loss(loss=loss_mean_relight, mode='vl', net='rel', step=epoch)
    return np.mean(losses_total)


def training_join(n_epochs, decom_lr, relight_lr, s_epochs,
                  ignore_ienhanace=False):
    train_data_loader, val_data_loader, _ \
        = MyDataLoader().get_data_loaders(path_low='/opt/proj_img_enhance/data/train/low',
                                          path_high='/opt/proj_img_enhance/data/train/high')

    model_decomposition = DecomNet().to(device)
    model_relight = RelightNet().to(device)

    optimizer = optim.Adam([{"params": model_decomposition.parameters(),
                             "lr": decom_lr},
                            {"params": model_relight.parameters(),
                             "lr": relight_lr}])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                     factor=0.5)

    date_dir = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    path_decomposition = os.path.join('checkpoints', 'join', 'decomposition',
                                      date_dir)
    os.mkdir(path_decomposition)
    path_relight = os.path.join('checkpoints', 'join', 'relight', date_dir)
    os.mkdir(path_relight)

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")

        print("Training")
        train(model_decomposition, model_relight, train_data_loader, optimizer,
              epoch, ignore_ienhanace)

        print("Validation")
        loss_mean_total = evaluation(model_decomposition, model_relight,
                                     val_data_loader, epoch, ignore_ienhanace)
        scheduler.step(loss_mean_total)
        logger.log_learning_rate(optimizer, net='join', step=epoch)

        if epoch % s_epochs == 0:
            savedir_decomposition = os.path.join(path_decomposition,
                                                 f"model_decomposition_epoch_{epoch}.pt")
            savedir_relight = os.path.join(path_relight,
                                           f"model_relight_epoch_{epoch}.pt")
            save_model(model_decomposition, optimizer, epoch,
                       savedir_decomposition)
            save_model(model_relight, optimizer, epoch, savedir_relight)
