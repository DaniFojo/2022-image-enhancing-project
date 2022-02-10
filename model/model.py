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

	def __init__(self, out_channels=64, krnl_size=3):
		super().__init__()
		

		# pytorch:	
		# 	self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

		# tensorflow:
		# en el código de Retinex:
		# 	conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same',...)
		# Ejemplo en stackoverflow:
		#	Convolution Layer with 32 filters and a kernel size of 5
		#	conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu) 
		# tf.layers.conv2d: 
		# 	filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
		#   (out channels)
		# https://www.tutorialexample.com/understand-tf-layers-conv2d-with-examples-tensorflow-tutorial/
		# aparentemente obtiene el in_channels de la propia imagen, asumiendo que es de la forma:
		#    [batch, in_height, in_width, in_channels]

		# Tensorflow Padding:
		# https://wandb.ai/krishamehta/seo/reports/Difference-Between-SAME-and-VALID-Padding-in-TensorFlow--VmlldzoxODkwMzE
		# When padding == ”VALID”, the input image is not padded.
		# When padding == “SAME”, the input is half padded. 
		# The padding type is called SAME because the output size is the same as the input size (when stride=1). 
		# Using ‘SAME’ ensures that the filter is applied to all the elements of the input. 

		# En pytorch debo calcular el padding para obtener output size = input size:
		# P = (K -1) / 2


		# Convolutional with no activation function:
		# Notar kernel size * 3.
		kern = krnl_size * 3
		padd = (kern -1) / 2
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=kern, stride=1, padding=padd)  

		# 5 convolutional layers with a ReLU 
		self.conv2 = nn.Conv2d(...)  # + relu
		self.conv3 = nn.Conv2d(...)  # + relu
		self.conv4 = nn.Conv2d(...)  # + relu
		self.conv5 = nn.Conv2d(...)  # + relu
		self.conv6 = nn.Conv2d(...)  # + relu

		# Convolutional with no activation function:
		self.conv7 = nn.Conv2d(...)  

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.conv7(x)

		# parte x en 2:
		R = F.sigmoid(x[:,:,:,0:3]) # pseudocódigo sin revisar.
		L = F.sigmoid(x[:,:,:,3:4]) # pseudocódigo sin revisar.

    	return R, L






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

