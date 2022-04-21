from torchvision.utils import make_grid
import wandb


class WandbLogger():
    def __init__(self):
        wandb.login()
        wandb.init(project="retinex")

    def make_grid_wandb(self, img, tag, idx=5):
        grid = make_grid(img[:idx])
        wdb_image = wandb.Image(grid, tag)
        return wdb_image

    def log_images_grid(self, img_low=None, img_high=None, i_low=None,
                        i_high=None, r_low=None, r_high=None,
                        i_enhanced=None, reconstructed=None,
                        mode='tr', net='decom', step=None):
        suf = f'{mode}_{net}'

        d_log = {}

        if img_low is not None:
            wdb_low = self.make_grid_wandb(img_low,
                                           f"{suf}_image_low")
            d_log[f"{suf}_image_low"] = wdb_low

        if img_high is not None:
            wdb_high = self.make_grid_wandb(img_high,
                                            f"{suf}_image_high")
            d_log[f"{suf}_image_high"] = wdb_high

        if i_low is not None:
            wdb_ilow = self.make_grid_wandb(i_low, f"{suf}_ilow")
            d_log[f"{suf}_ilow"] = wdb_ilow

        if i_high is not None:
            wdb_ihigh = self.make_grid_wandb(i_high, f"{suf}_ihigh")
            d_log[f"{suf}_ihigh"] = wdb_ihigh

        if r_low is not None:
            wdb_rlow = self.make_grid_wandb(r_low, f"{suf}_rlow")
            d_log[f"{suf}_rlow"] = wdb_rlow

        if r_high is not None:
            wdb_rhigh = self.make_grid_wandb(r_high, f"{suf}_rhigh")
            d_log[f"{suf}_rhigh"] = wdb_rhigh

        if i_enhanced is not None:
            wdb_ienhanced = self.make_grid_wandb(i_enhanced,
                                                 f"{suf}_ienhanced")
            d_log[f"{suf}_ienhanced"] = wdb_ienhanced

        if reconstructed is not None:
            wdb_reconstructed = self.make_grid_wandb(reconstructed,
                                                     f"{suf}_reconstructed")
            d_log[f"{suf}_reconstructed"] = wdb_reconstructed

        wandb.log(d_log, step=step)

    def log_loss(self, loss, mode, net, step):
        wandb.log({f"loss_{mode}_{net}": loss}, step=step)

    def log_learning_rate(self, opt, net, step):
        learning_rate = opt.param_groups[0]['lr']
        wandb.log({f"learning_rate_{net}": learning_rate}, step=step)
