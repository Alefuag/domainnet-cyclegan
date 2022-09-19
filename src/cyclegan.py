

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, add, Conv2DTranspose, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.activations import tanh
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow_addons.layers import InstanceNormalization

# FID Score
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm


# To do:
# resnet-params by command line

# Done:
# integrate in CycleGAN class
# filter size in unet


###	generator F: domain X -> domain Y
###	generator G: domain Y -> domain X
### discriminator X: domain X -> output
### discriminator Y: domain Y -> output


### Vars ###
image_size = (256, 256, 3)
batch_size = 32

kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
gamma_initializer = RandomNormal(mean=0.0, stddev=0.02)


### Setup functions ###

# image_size = (256, 256, 3)
def set_image_size(size):
	global image_size
	image_size = size

# batch size = 32
def set_batch_size(size):
	global batch_size
	batch_size = size


### Losses and metrics ###

adversarial_loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def discriminator_loss(real, generated, loss_fn=adversarial_loss_fn):
	real_loss = loss_fn(tf.ones_like(real), real)
	generated_loss = loss_fn(tf.zeros_like(generated), generated)
	total_loss = (real_loss + generated_loss) * 0.5
	return total_loss

@tf.function
def generator_loss(generated, loss_fn=adversarial_loss_fn):
	generated_loss = loss_fn(tf.ones_like(generated), generated)
	return generated_loss

def get_default_optimizer(lr=2e-4, beta_1=0.5):
	return Adam(learning_rate=lr, beta_1=beta_1)


### Discriminator ###

def build_discriminator(image_shape=image_size, filters=64, k_init=kernel_initializer, name=None):

	# Input keras layer
	input_image = Input(shape=image_shape)
	# Conv1 block
	d = Conv2D(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(input_image)
	d = LeakyReLU(alpha=0.2)(d)
	# Conv2 block
	d = Conv2D(filters*2, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# Conv3 block
	d = Conv2D(filters*4, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# Conv4 block
	d = Conv2D(filters*8, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# Conv5 block
	d = Conv2D(filters*8, (4,4), padding='same', kernel_initializer=k_init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# output layer
	output_layer = Conv2D(1, (4,4), padding='same', kernel_initializer=k_init)(d)
	
	# build model
	model = Model(input_image, output_layer, name=name)
	#### compile model
	#### model.compile(loss='functionloss', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


### Building blocks for the ResNet generator ###

class ReflectionPadding2D(tf.keras.layers.Layer):
	'''layer used instead of zero padding for conv layers'''
	def __init__(self, padding=(1, 1), **kwargs):
		self.padding = padding
		super(ReflectionPadding2D, self).__init__(**kwargs)

	def call(self, in_tensor, mask=None):
		padding_width, padding_height = self.padding
		# padding only width and height, not in sample and channel dimension
		padding_tensor = [
            [0, 0],
			[padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
		]
		return tf.pad(in_tensor, padding_tensor, mode="REFLECT")


def resnet_block(in_tensor, act, k_init=kernel_initializer, k_size=(3,3), strides=(1,1), padding='valid', g_init=gamma_initializer, use_bias=False):

	channels = in_tensor.shape[-1]
	r = in_tensor

	r = ReflectionPadding2D()(r)
	r = Conv2D(channels, k_size, strides=strides, padding=padding, kernel_initializer=k_init, use_bias=use_bias)(r)
	r = InstanceNormalization(gamma_initializer=g_init)(r)
	r = act(r)

	r = ReflectionPadding2D()(r)
	r = Conv2D(channels, k_size, strides=strides, padding=padding, kernel_initializer=k_init, use_bias=use_bias)(r)
	r = InstanceNormalization(gamma_initializer=g_init)(r)
	
	r = add([r, in_tensor])

	return r

def downsample(input_tensor, filters, activation, k_init=kernel_initializer, k_size=(3,3), strides=(2,2), padding='same', g_init=gamma_initializer, use_bias=False):
	d = input_tensor
	
	d = Conv2D(
		filters,
		k_size,
		strides=strides,
		padding=padding,
		kernel_initializer=k_init,
		use_bias=use_bias
	)(d)

	d = InstanceNormalization(gamma_initializer=g_init)(d)

	if activation:
		d = activation(d)
	
	return d

def upsample(input_tensor, filters, activation, k_init=kernel_initializer, k_size=(3,3), strides=(2,2), padding='same', g_init=gamma_initializer, use_bias=False):
	u = input_tensor
	u = Conv2DTranspose(
		filters,
		k_size,
		strides=strides,
		padding=padding,
		kernel_initializer=k_init,
		use_bias=use_bias
	)(u)

	u = InstanceNormalization(gamma_initializer=g_init)(u)

	if activation:
		u = activation(u)
	
	return u


### ResNet generator ###

def build_resnet(
	image_shape=image_size,
	filters=64,
	down_blocks=2,
	res_blocks=9,
	up_blocks=2,
	k_init=kernel_initializer,
	g_init=gamma_initializer,
	name=None):
	
	input_image = Input(shape=image_shape)
	# Amplify image to maintain the same resolution after Conv2D
	r = ReflectionPadding2D(padding=(3,3))(input_image)
	# Apply first Conv2D layer
	r = Conv2D(filters, (7,7) )(r)
	r = InstanceNormalization(gamma_initializer=g_init)(r)
	r = ReLU()(r)

	# Start downsampling
	for i in range(down_blocks):
		filters *= 2
		r = downsample(r, filters, ReLU())
	
	# Start residual blocks
	for i in range(res_blocks):
		r = resnet_block(r, ReLU())

	# Start upsampling
	for i in range(up_blocks):
		filters //= 2
		r = upsample(r, filters, ReLU())
	
	# Output layer
	r = ReflectionPadding2D(padding=(3,3))(r)
	# 3 channels for RGB image
	r = Conv2D(3, (7,7) )(r)

	output_image = tanh(r)

	model = Model(input_image, output_image, name=name)

	return model



def build_unet(image_shape=image_size, unet_depth=8, k_init=RandomNormal(stddev=0.02), name=None):
	'''
	max depth of the U-Net is 8, minimum recommended is 4
	'''

	# output_channels = 3

	down_filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512]
	
	# check depth of U-Net
	if unet_depth < len(down_filters):
		down_filters = down_filters[:unet_depth]

	# reverse down_filters to get up_filters
	up_filters = down_filters[:-1]
	up_filters.reverse()
	
	skip_connections = []

	input_image = Input(shape=image_shape)
	u = input_image

	# downsample
	for down_filter in down_filters:
		u = downsample(u, down_filter, ReLU(), k_init)
		skip_connections.append(u)

	# Check down and up filters
	# print('down_filters:', down_filters)
	# print('up_filters:', up_filters)


	# remove last downsample output from skip connections
	skip_connections.pop()
	
	# upsample
	for up_filter in up_filters:
		u = upsample(u, up_filter, ReLU(), k_init)
		u = Concatenate()([ u, skip_connections.pop() ])

	# last layer
	u = Conv2DTranspose(
		3, # 3 channels for RGB image
		4,
		strides=2,
		padding='same',
		kernel_initializer=k_init,
		activation='tanh')(u)  # (bs, 256, 256, 3)

	output_image = tanh(u)
	
	model = Model(inputs=input_image, outputs=output_image, name=name)

	return model


def build_generator(model = 'resnet', *args, **kwargs):
	if model == 'resnet':
		return build_resnet(*args, **kwargs)
	elif model == 'unet':
		return build_unet(*args, **kwargs)
	else:
		raise ValueError('model must be either resnet or unet')


class FID_Score():
	def __init__(self, real_X, real_Y):
		self.inception = InceptionV3(include_top=False, input_shape=image_size, pooling='avg')
		features_real_X = self.inception.predict(real_X)
		features_real_Y = self.inception.predict(real_Y)

		# mean and covariance of domain X images
		self.mu_X = features_real_X.mean(axis=0)
		self.sigma_X = cov(features_real_X, rowvar=False)

		# mean and covariance of domain Y images
		self.mu_Y = features_real_Y.mean(axis=0)
		self.sigma_Y = cov(features_real_Y, rowvar=False)


	
	def __call__(self, fake_X, fake_Y):
		features_fake_X = self.inception.predict(fake_X)
		features_fake_Y = self.inception.predict(fake_Y)

		mu_fake_X, sigma_fake_X = features_fake_X.mean(axis=0), cov(features_fake_X, rowvar=False)
		ssdiff_X = np.sum((self.mu_X - mu_fake_X)**2.0)
		covmean_X = sqrtm(self.sigma_X.dot(sigma_fake_X))
		if iscomplexobj(covmean_X):
			covmean_X = covmean_X.real
		fid_X = ssdiff_X + trace(self.sigma_X + sigma_fake_X - 2.0 * covmean_X)

		mu_fake_Y, sigma_fake_Y = features_fake_Y.mean(axis=0), cov(features_fake_Y, rowvar=False)
		ssdiff_Y = np.sum((self.mu_Y - mu_fake_Y)**2.0)
		covmean_Y = sqrtm(self.sigma_Y.dot(sigma_fake_Y))
		if iscomplexobj(covmean_Y):
			covmean_Y = covmean_Y.real
		fid_Y = ssdiff_Y + trace(self.sigma_Y + sigma_fake_Y - 2.0 * covmean_Y)

		return fid_X, fid_Y



class CycleGAN(tf.keras.models.Model):

	def __init__(
		self,
		gen_model='resnet',
		net_params={},
		resnet_params={}, #{'filters':64, 'down_blocks':2, 'res_blocks':9, 'up_blocks':2},
		unet_params={}, #{'filters':64, 'down_blocks':2, 'up_blocks':2},
		cycle_weight = 10.0,
		identity_weight = 0.5,
		fid = None
	):
		super(CycleGAN, self).__init__()
		self.gen_G = build_generator(model=gen_model, name='gen_G', **net_params)
		self.gen_F = build_generator(model=gen_model, name='gen_F', **net_params)
		self.disc_X = build_discriminator(name='disc_X')
		self.disc_Y = build_discriminator(name='disc_Y')
		self.cycle_weight = cycle_weight
		self.identity_weight = identity_weight
		self.fid = fid
		self.__call__(tf.zeros( (2, 1, *image_size) ), training=False) # initialize model


	@tf.function
	def call(self, inputs):
		X = inputs[0]
		Y = inputs[1]
		return (self.gen_G(X), self.gen_F(Y))

	def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss,
        disc_loss,
		cycle_loss=tf.keras.losses.MeanAbsoluteError(),
		identity_loss=tf.keras.losses.MeanAbsoluteError()
    ):
		super(CycleGAN, self).compile()
		self.gen_G_optimizer = gen_G_optimizer
		self.gen_F_optimizer = gen_F_optimizer
		self.disc_X_optimizer = disc_X_optimizer
		self.disc_Y_optimizer = disc_Y_optimizer
		self.gen_loss_metric = gen_loss
		self.disc_loss_metric = disc_loss
		self.cycle_loss_metric = cycle_loss
		self.identity_loss_metric = identity_loss

	@tf.function
	def train_step(self, dataset_batch):
		dom_X, dom_Y = dataset_batch
		
		# print('dom_X shape:', dom_X.shape, 'dom_Y shape:', dom_Y.shape)

		with tf.GradientTape(persistent=True) as tape:

			# Generate translated images
			fake_Y = self.gen_G(dom_X, training=True)
			fake_X = self.gen_F(dom_Y, training=True)

			# Generate reconstructed images
			cycled_X = self.gen_F(fake_Y, training=True)
			cycled_Y = self.gen_G(fake_X, training=True)

			# Generate identity images
			identity_X = self.gen_F(dom_X, training=True)
			identity_Y = self.gen_G(dom_Y, training=True)

			# Classify real images
			disc_real_X = self.disc_X(dom_X, training=True)
			disc_real_Y = self.disc_Y(dom_Y, training=True)

			# Classify fake images
			disc_fake_X = self.disc_X(fake_X, training=True)
			disc_fake_Y = self.disc_Y(fake_Y, training=True)

			# Do not need to classify identity images, as they are only used for cycle loss

			# Simple generator losses
			gen_G_loss = self.gen_loss_metric(disc_fake_Y)
			gen_F_loss = self.gen_loss_metric(disc_fake_X)

			# Cycle loss
			cycle_G_loss = self.cycle_loss_metric(dom_Y, cycled_Y) * self.cycle_weight
			cycle_Y_loss = self.cycle_loss_metric(dom_X, cycled_X) * self.cycle_weight

			# Identity loss
			identity_G_loss = self.identity_loss_metric(dom_X, identity_X) * self.cycle_weight * self.identity_weight
			identity_F_loss = self.identity_loss_metric(dom_Y, identity_Y) * self.cycle_weight * self.identity_weight

			# Combined generator losses
			loss_G = gen_G_loss + cycle_G_loss + identity_G_loss
			loss_F = gen_F_loss + cycle_Y_loss + identity_F_loss

			# Discriminator losses
			disc_X_loss = self.disc_loss_metric(disc_real_X, disc_fake_X)
			disc_Y_loss = self.disc_loss_metric(disc_real_Y, disc_fake_Y)

		# Generator gradients based on losses
		gradients_gen_G = tape.gradient(loss_G, self.gen_G.trainable_variables)
		gradients_gen_F = tape.gradient(loss_F, self.gen_F.trainable_variables)

		# Discriminator gradients based on losses
		gradients_disc_X = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
		gradients_disc_Y = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

		# Update generator weights
		self.gen_G_optimizer.apply_gradients(zip(gradients_gen_G, self.gen_G.trainable_variables))
		self.gen_F_optimizer.apply_gradients(zip(gradients_gen_F, self.gen_F.trainable_variables))

		# Update discriminator weights
		self.disc_X_optimizer.apply_gradients(zip(gradients_disc_X, self.disc_X.trainable_variables))
		self.disc_Y_optimizer.apply_gradients(zip(gradients_disc_Y, self.disc_Y.trainable_variables))

		# fid_score_X, fid_score_Y = self.fid(fake_X, fake_Y) if self.fid else (0, 0)

		losses = {
			'gen_G': loss_G,
			'gen_F': loss_F,
			'generators': (loss_G + loss_F) * 0.5,
			'disc_X': disc_X_loss,
			'disc_Y': disc_Y_loss,
			'discriminators': (disc_X_loss + disc_Y_loss) * 0.5
			
		}

		return losses
		

