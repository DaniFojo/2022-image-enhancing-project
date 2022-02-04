from configparser import Interpolation
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
	"""
	Convolutional NN
	Takes in paired low/normal-light images and learns the decomposition for both.
	Constraint: images share the same reflectance.
	"""

    def __init__(self):
        super().__init__()
		
		# Convolutional with no activation function:
		self.conv1 = nn.Conv2d(...)  

		# 5 convolutional layers with a ReLU 
		self.conv2 = nn.Conv2d(...)  # + relu
		self.conv3 = nn.Conv2d(...)  # + relu
		self.conv4 = nn.Conv2d(...)  # + relu
		self.conv5 = nn.Conv2d(...)  # + relu
		self.conv6 = nn.Conv2d(...)  # + relu

		# Convolutional with no activation function:
		self.conv7 = nn.Conv2d(...)  

    def forward(self, x):
        ...




class RelightNet(nn.Module):
	"""
	Encoder-decoder with skip connections + denoising operation.
	"""

    def __init__(self):
        super().__init__()

		# Encoder
		self.conv1 = nn.Conv2d(...)  # + relu
		self.conv2 = nn.Conv2d(...)  # + relu
		self.conv3 = nn.Conv2d(...)  # + relu
		self.conv4 = nn.Conv2d(...)  # + relu


		# Decoder
		# "Resize-convolutional layer" consists of a nearest-neighbor
		# interpolation operation, a convolutional layer with stride 1, and a ReLU

		# No usa deconvolution (nn.ConvTranspose2d) ?

		up1 = upsample nearest neighbour Interpolation
		self.deconv1 = nn.Conv2d(...)  # + relu
		up2 = upsample nearest neighbour Interpolation
		self.deconv2 = nn.Conv2d(...)  # + relu
		up3 = upsample nearest neighbour Interpolation
		self.deconv3 = nn.Conv2d(...)  # + relu

		# Concat
 		deconv1_resize = image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])


		# 1x1 conv
        feature_fusion = conv2d(feature_gather, channel, 1, padding='same', activation=None)

		# conv 3x3
        output = conv2d(feature_fusion, 1, 3, padding='same', activation=None)



    def forward(self, x):
        ...



class ImageEnhance()

	def __init__(self):
 		[R_low, I_low] = DecomNet(self.input_low)
        [R_high, I_high] = DecomNet(self.input_high)
        
        I_delta = RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])
		# ...

		# Loss
		# ...

		# Optimizer
		# ...

