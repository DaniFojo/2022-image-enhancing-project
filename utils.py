import cv2
from torchvision import transforms
import numpy as np
import os
import torch
from model.model import DecomNet, RelightNet


def denoise(torch_image):
    trans = transforms.ToPILImage()
    cv2_image = np.array(trans(torch_image))
    denoised_image = cv2.fastNlMeansDenoisingColored(cv2_image, None, 10, 10, 7, 21)


def create_directory_if_not_exists(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def create_directories():
    create_directory_if_not_exists('checkpoints')
    create_directory_if_not_exists(os.path.join('checkpoints','join'))
    create_directory_if_not_exists(os.path.join('checkpoints','join','decomposition'))
    create_directory_if_not_exists(os.path.join('checkpoints','join','relight'))
    create_directory_if_not_exists(os.path.join('checkpoints','split'))
    create_directory_if_not_exists(os.path.join('checkpoints','split','decomposition'))
    create_directory_if_not_exists(os.path.join('checkpoints','split','relight'))
    

def save_model(model, optimizer, epoch, savedir):
    print(f"Saving checkpoint to {savedir}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, savedir)


def load_model(checkpoint_path, model_name, device):
    checkpoint = torch.load(checkpoint_path)
    if model_name == 'decom':
        model = DecomNet().to(device)
    else:
        model = RelightNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model        