import os
import torch
import datetime

from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger():

    def __init__(self, task, model):
        #logdir = os.path.join("logs", f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        #self.writer = SummaryWriter(log_dir=logdir)
        pass
        
        
    def log_training(self, model, epoch, train_loss_avg, val_loss_avg):
        # self.writer...
        pass
