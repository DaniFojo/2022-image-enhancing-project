import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecomNet(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3):
        super().__init__()
        # Convolutional with no activation function:
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')

        # 5 convolutional layers with a ReLU
        self.conv1 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')
        self.conv5 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')

        # Convolutional with no activation function:
        self.conv6 = nn.Conv2d(in_channels=out_channels,
                               out_channels=4,
                               kernel_size=kernel_size,
                               stride=1, padding='same')

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        # Reflectance and illuminance
        reflectance = torch.sigmoid(x[:, 0:3, :, :])
        illumination = torch.sigmoid(x[:, 3:4, :, :])
        return reflectance, illumination


class RelightNet(nn.Module):
    """
    Encoder-decoder with skip connections + denoising operation.
    """

    def __init__(self, out_channels=64, kernel_size=3):
        super().__init__()
        self.padding = int((kernel_size - 1) / 2)

        # Encoder
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1, padding='same')
        self.conv1 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=2, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=2, padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=2, padding=self.padding)

        # Decoder
        self.deconv1 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=1, padding='same')
        self.deconv2 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=1, padding='same')
        self.deconv3 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=1, padding='same')

        # Last Convolutions
        self.convO = nn.Conv2d(in_channels=3 * out_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1,
                               padding='same')
        self.convF = nn.Conv2d(in_channels=out_channels,
                               out_channels=1,
                               kernel_size=kernel_size,
                               stride=1, padding='same')

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.interpolate(input=x3, size=(x2.shape[2], x2.shape[3]),
                           mode='bicubic')
        x4 = F.relu(self.deconv1(x4)) + x2
        x5 = F.interpolate(input=x4, size=(x1.shape[2], x1.shape[3]),
                           mode='bicubic')
        x5 = F.relu(self.deconv2(x5)) + x1
        x6 = F.interpolate(input=x5, size=(x0.shape[2], x0.shape[3]),
                           mode='bicubic')
        x6 = F.relu(self.deconv3(x6)) + x0
        x7 = F.interpolate(input=x4, size=(x6.shape[2], x6.shape[3]),
                           mode='bicubic')
        x8 = F.interpolate(input=x5, size=(x6.shape[2], x6.shape[3]),
                           mode='bicubic')
        x = torch.cat((x6, x7, x8), dim=1)
        x = self.convO(x)
        x = self.convF(x)
        return x


class RelightNetConvTrans(nn.Module):
    """
    Encoder-decoder with skip connections + denoising operation.
    """
    def __init__(self, out_channels=64, kernel_size=3):
        super().__init__()
        self.padding = int((kernel_size - 1) / 2)

        # Encoder
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding='same')
        self.conv1 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=2,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=2,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=2,
                               padding=self.padding)
        self.deconv1 = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=2, padding=1)
        self.convO = nn.Conv2d(in_channels=3 * out_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding='same')
        self.convF = nn.Conv2d(in_channels=out_channels, out_channels=1,
                               kernel_size=kernel_size,
                               stride=1, padding='same')

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.deconv1(x3)) + x2
        x5 = F.relu(self.deconv2(x4)) + x1
        x6 = F.relu(self.deconv3(x5)) + x0
        x7 = F.interpolate(input=x4, size=(x6.shape[2], x6.shape[3]),
                           mode='bicubic')
        x8 = F.interpolate(input=x5, size=(x6.shape[2], x6.shape[3]),
                           mode='bicubic')
        x = torch.cat((x6, x7, x8), dim=1)
        x = self.convO(x)
        x = self.convF(x)
        return x


def smooth(r, i):
    r = tv.transforms.functional.rgb_to_grayscale(r)
    gradient_x = gradient(i, "x") * torch.exp(-10 * ave_gradient(r, "x"))
    gradient_y = gradient(i, "y") * torch.exp(-10 * ave_gradient(r, "y"))
    gradient_avg = torch.mean(gradient_x + gradient_y)
    return gradient_avg


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]],
                                    dtype=torch.float32),
                                    (1, 1, 2, 2)).to(device)
    smooth_kernel_y = torch.transpose(smooth_kernel_x,
                                      dim0=2, dim1=3).to(device)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad = torch.abs(F.conv2d(input=input_tensor, weight=kernel, stride=1,
                              padding='same'))
    return grad


def ave_gradient(input_tensor, direction):
    avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    avg_grad = avg_pool(gradient(input_tensor, direction))
    return avg_grad


def loss_decom_net(input_low, input_high, r_low, i_low, r_high, i_high):
    i_low3 = torch.concat((i_low, i_low, i_low), dim=1)
    i_high3 = torch.concat((i_high, i_high, i_high), dim=1)
    loss_recon_low = torch.mean(torch.abs(input_low - r_low * i_low3))
    loss_recon_high = torch.mean(torch.abs(input_high - r_high * i_high3))
    loss_recon_mutal_low = torch.mean(torch.abs(input_low - r_high * i_low3))
    loss_recon_mutal_high = torch.mean(torch.abs(input_high - r_low * i_high3))
    loss_invariable_reflectance = torch.mean(torch.abs(r_low - r_high))
    loss_illumination_smoothness_low = smooth(r_low, i_low)
    loss_illumination_smoothness_high = smooth(r_high, i_high)
    loss_decom = loss_recon_low + loss_recon_high + 0.001 * \
        loss_recon_mutal_low + 0.001 * \
        loss_recon_mutal_high + 0.01 * \
        loss_invariable_reflectance + 0.1 * \
        loss_illumination_smoothness_low + 0.1 * \
        loss_illumination_smoothness_high
    return loss_decom


def loss_relight_net(input_high, r_low, i_enhanced):
    i_enhanced3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
    loss_recon = torch.mean(torch.abs(input_high - r_low * i_enhanced3))
    loss_illumination_smoothness = smooth(r_low, i_enhanced)
    loss_relight = loss_recon + 3 * loss_illumination_smoothness
    return loss_relight
