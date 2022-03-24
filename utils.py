import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Aux functions

def compute_accuracy(preds, labels):
    pred_labels = preds.argmax(dim=1, keepdim=True)
    acc = pred_labels.eq(labels.view_as(pred_labels)).sum().item() / len(
        pred_labels)
    return acc

def denoise(torch_image):
    trans = transforms.ToPILImage()
    cv2_image = np.array(trans(torch_image))
    denoised_image = cv2.fastNlMeansDenoisingColored(cv2_image, None, 10, 10, 7, 21)
    plt.imshow(denoised_image)
    plt.show()