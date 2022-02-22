import datetime

import numpy as np
import pandas as pd

import torch
import wandb

###
# logger = WandbLogger(args.task, model)


PROJECT_NAME = 'image-enhancing'

class WandbLogger():

    def __init__(self, task, model):
        wandb.login()
        wandb.init(project=PROJECT_NAME)
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch
        wandb.watch(model, log='all', log_freq=100)

    def log_classification_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, val_acc_avg, train_acc_avg):
        # Log confusion matrix figure to wandb
        # wandb.log({'Confusion Matrix': fig,
        #            "epoch": epoch
        #            })

        #  Log validation loss to wandb
        wandb.log({PROJECT_NAME+'/val_loss': val_loss_avg,
                   'epoch': epoch
                   })

        # Log validation accuracy to wandb
        wandb.log({PROJECT_NAME+'/val_acc': val_acc_avg,
                   'epoch': epoch
                   })
        # Log training loss to wandb
        wandb.log({PROJECT_NAME+'train_loss': train_loss_avg,
                   'epoch': epoch
                   })

        # Log train accuracy to wandb
        wandb.log({PROJECT_NAME+'/train_acc': train_acc_avg,
                   'epoch': epoch
                   })

    def log_embeddings(self, model, train_loader, device):
        out = model.encoder.linear.out_features
        columns = np.arange(out).astype(str).tolist()
        columns.insert(0, "target")
        columns.insert(0, "image")

        list_dfs = []

        for i in range(3): # take only 3 batches of data for plotting
            images, labels = next(iter(train_loader))

            for img, label in zip(images, labels):
                # forward img through the encoder
                image = wandb.Image(img)
                label = label.item()
                latent = model.encoder(img.to(device).unsqueeze(dim=0)).squeeze().detach().cpu().numpy().tolist()
                data = [image, label, *latent]

                df = pd.DataFrame([data], columns=columns)
                list_dfs.append(df)
        embeddings = pd.concat(list_dfs, ignore_index=True)

        # Log latent representations (embeddings)
        wandb.log({"mnist": embeddings})

    def log_model_graph(self, model, train_loader):
        # Wandb does not support logging the model graph
        pass