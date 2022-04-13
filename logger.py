import datetime
import torch
import wandb
from torchvision.utils import make_grid


class WandbLogger():
    def __init__(self, task = ""):
        wandb.login()
        # wandb.init(project="retinex", mode="disabled") # disable wandb logging.
        wandb.init(project="retinex")  # añadir hyperparameters en el init de wandb
        # Por si queremos añadir nombre y fecha a la tarea
        # wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    def make_grid_wandb(self, img, tag, idx=5):
        grid = make_grid(img[:idx])
        wdb_image = wandb.Image(grid, tag)
        return wdb_image

    def log_images_grid(self, img_low=None, img_high=None, i_low=None, i_high=None, r_low=None, r_high=None, i_enhanced=None, reconstructed=None, mode='tr', net='decom', step=None):
        suf = f'{mode}_{net}'

        d = {}

        if img_low is not None:
            wdb_low = self.make_grid_wandb(img_low, f"{suf}_image_low")
            d[f"{suf}_image_low"] = wdb_low

        if img_high is not None:
            wdb_high = self.make_grid_wandb(img_high, f"{suf}_image_high")
            d[f"{suf}_image_high"] = wdb_high

        if i_low is not None:
            wdb_ilow = self.make_grid_wandb(i_low, f"{suf}_ilow")
            d[f"{suf}_ilow"] = wdb_ilow

        if i_high is not None:
            wdb_ihigh = self.make_grid_wandb(i_high, f"{suf}_ihigh")
            d[f"{suf}_ihigh"] = wdb_ihigh

        if r_low is not None:
            wdb_rlow = self.make_grid_wandb(r_low, f"{suf}_rlow")
            d[f"{suf}_rlow"] = wdb_rlow

        if r_high is not None:
            wdb_rhigh = self.make_grid_wandb(r_high, f"{suf}_rhigh")
            d[f"{suf}_rhigh"] = wdb_rhigh

        if i_enhanced is not None:
            wdb_ienhanced = self.make_grid_wandb(i_enhanced, f"{suf}_ienhanced")
            d[f"{suf}_ienhanced"] = wdb_ienhanced
        
        if reconstructed is not None:
            wdb_reconstructed = self.make_grid_wandb(reconstructed, f"{suf}_reconstructed")
            d[f"{suf}_reconstructed"] = wdb_reconstructed

        wandb.log(d, step=step)


    def log_loss(self, loss, mode, net, step):
        wandb.log({f"loss_{mode}_{net}": loss}, step=step)

    def log_learning_rate(self, opt, net, step):
        lr = opt.param_groups[0]['lr']
        wandb.log({f"learning_rate_{net}": lr}, step=step)
