import datetime
import torch
import wandb
from torchvision.utils import make_grid


class WandbLogger():
    def __init__(self, task = ""):
        wandb.login()
        wandb.init(project="retinex")  # añadir hyperparameters en el init de wandb
        # Por si queremos añadir nombre y fecha a la tarea
        # wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    def make_grid_wandb(self, img, tag, idx=5):
        grid = make_grid(img[:idx])
        wdb_image = wandb.Image(grid, tag)
        return wdb_image

    def log_images_grid(self, img_low, img_high, i_low, i_high, r_low, r_high, i_enhanced, reconstructed, mode='train'):
        if mode == 'train':
            suf = 'tr'
        elif mode == 'validation':
            suf = 'vl'
        wdb_low = self.make_grid_wandb(img_low, f"{suf}_input_image_low")
        wdb_high = self.make_grid_wandb(img_high, f"{suf}_input_image_high")
        wdb_ilow = self.make_grid_wandb(i_low, f"{suf}_ilow")
        wdb_ihigh = self.make_grid_wandb(i_high, f"{suf}_ihigh")
        wdb_rlow = self.make_grid_wandb(r_low, f"{suf}_rlow")
        wdb_rhigh = self.make_grid_wandb(r_high, f"{suf}_rhigh")
        wdb_ienhanced = self.make_grid_wandb(i_enhanced, f"{suf}_ienhanced")
        wdb_reconstructed = self.make_grid_wandb(reconstructed, f"{suf}_reconstructed")
        wandb.log({f"{suf}_image_low": wdb_low, 
                   f"{suf}_image_high": wdb_high, 
                   f"{suf}_ilow": wdb_ilow,
                   f"{suf}_ihigh": wdb_ihigh, 
                   f"{suf}_rlow": wdb_rlow, 
                   f"{suf}_rhigh": wdb_rhigh, 
                   f"{suf}_ienhanced": wdb_ienhanced,
                   f"{suf}_reconstructed": wdb_reconstructed})


    def log_images_grid_asPaper(self, img_low, img_high, i_low, i_norm, r_low, r_high, i_enhanced):
        aux_1 = torch.full((3,300, 300), 1)
        aux_2 = torch.full((3,150, 300), 1)
        col1 = torch.cat((aux_2, img_high, aux_1, img_low, aux_2), 1)
        col2 = torch.cat((r_high, i_norm, r_low, i_low), 1)
        col3 = torch.cat((aux_1, aux_1, aux_2, i_enhanced, aux_2), 1)
        grid = torch.cat((col1, col2, col3), 2)
        wdb_grid = wandb.Image(grid, "batch_example")
        wandb.log({"batch_example": wdb_grid})

    def log_training(self, loss_total, loss_decom, loss_relight) :
        wandb.log({"loss_total_train": loss_total, "loss_decom_train": loss_decom, "loss_relight_train": loss_relight})

    def log_eval(self, loss_total, loss_decom, loss_relight) :
        wandb.log({"loss_total_eval": loss_total, "loss_decom_eval": loss_decom, "loss_relight_eval": loss_relight})