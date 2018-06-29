"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).

To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.

The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
import time
import cv2
import argparse
import os
import numpy as np
import keras
import keras.layers as layers
from train_gen import Gen
from keras.models import *
from keras.layers import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')
    exit()
image_shape=(256,256,3)
img_size=256
BATCH_SIZE = 5
TRAINING_RATIO = 2  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
gen = Gen('images/kak_ygodno.txt', 'images/X/', 'images/new_Y/','Y_validation', 1, img_size)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    concat_axis = 3
    inputs = layers.Input(shape = image_shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
    up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
    up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis) 
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
    up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
    up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = layers.Conv2D(3, (1, 1))(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    # model = Sequential()
    # model.add(Dense(1024, input_dim=100))
    # model.add(LeakyReLU())
    # model.add(Dense(128 * 7 * 7))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    # if K.image_data_format() == 'channels_first':
    #     model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    #     bn_axis = 1
    # else:
    #     model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    #     bn_axis = -1
    # model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Convolution2D(64, (5, 5), padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # # Because we normalized training inputs to lie in the range [-1, 1],
    # # the tanh function should be used for the output of the generator to ensure its output
    # # also lies in this range.
    # model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.

    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    inputs = layers.Input(shape = image_shape)
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        x=Convolution2D(64, (5, 5), padding='same', input_shape=(3, 512, 512),name="our_conv_1")(inputs)
    else:
        x=Convolution2D(64, (5, 5), padding='same', input_shape=(512, 512, 3),name="our_conv_1")(inputs)
    x=Convolution2D(64, (5, 5), kernel_initializer='he_normal', strides=[2, 2],name="our_conv_2")(x)
    x=MaxPooling2D(pool_size=2,name="our_pool_1")(x)
    x=LeakyReLU()(x)
    x=Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2],name="our_conv_3")(x)
    x=LeakyReLU()(x)
    x=Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2],name="our_conv_4")(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D(pool_size=2,name="our_pool_2")(x)
    x=Convolution2D(256, (3, 3), kernel_initializer='he_normal', strides=[2, 2],name="our_conv_5")(x)
    x=LeakyReLU()(x) 
    x=Convolution2D(256, (2, 2), kernel_initializer='he_normal', strides=[2, 2],name="our_conv_6")(x)
    x=LeakyReLU()(x)   
    x=Dropout(0.5)(x)
    # model.add(MaxPooling2D(pool_size=2,name="our_pool_3"))
    x=Flatten()(x)   
    x=Dense(256, kernel_initializer='he_normal',name="our_dense_1")(x)
    x=LeakyReLU()(x)
    x=Dense(1, kernel_initializer='he_normal',name="our_dense_2")(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    image_y=np.array(gen.get_test())
    # print(image_y.shape)
    test_image_stack = generator_model.predict(image_y)
    test_image_stack = (test_image_stack * 127.5) + 127.5
     
    for z in range(len(test_image_stack)):
        # test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
        #tiled_output = tile_images(test_image_stack)
        tiled_output= np.concatenate((image_y[z],test_image_stack[z]), axis=1)
        #tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
        outfile = os.path.join(output_dir, 'epoch_{}_{}.png'.format(epoch,z))
        cv2.imwrite(outfile,tiled_output)
        #tiled_output.save(outfile)


parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated files to")
args = parser.parse_args()

# First we load the image data, reshape it and normalize it to the range [-1, 1]
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train = np.concatenate((X_train, X_test), axis=0)
# if K.image_data_format() == 'channels_first':
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
# else:
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

# X_train, y_train = gen.get_epoch(BATCH_SIZE)

# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print(X_train.shape)
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# y_train = (y_train.astype(np.float32) - 127.5) / 127.5

# Now we initialize the generator and discriminator.
generator = make_generator()
discriminator = make_discriminator()
# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=image_shape)
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=image_shape)
generator_input_for_discriminator = Input(shape=image_shape)
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
minibatches_size = BATCH_SIZE * TRAINING_RATIO
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
# positive_y_generator_train = np.ones((minibatches_size, 1), dtype=np.float32)
for epoch in range(1000):
    #np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    #print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    cur_time=time.time()
    for i in range(int(len(gen.files) // (BATCH_SIZE * TRAINING_RATIO))):
        X_train, y_train = gen.get_epoch(minibatches_size)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        y_train = (y_train.astype(np.float32) - 127.5) / 127.5
        # print("X_train.shape")
        # print(X_train.shape,y_train.shape)
        # discriminator_minibatches_x = X_train[0:(i + 1) * minibatches_size]
        # discriminator_minibatches_y = y_train[i * minibatches_size:(i + 1) * minibatches_size]
        # print("discriminator_minibatches_x.shape")
        # print(discriminator_minibatches_x.shape,discriminator_minibatches_y.shape)
        for j in range(TRAINING_RATIO):
            image_batch = X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            image_batch_y = y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            #noise = np.random.rand(BATCH_SIZE, 512,512,3).astype(np.float32)
            # print(noise.shape)
            # print("image_batch.shape")
            # print(image_batch.shape,image_batch_y.shape,positive_y.shape,negative_y.shape,dummy_y.shape)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch_y, image_batch],
                                                                         [positive_y, negative_y, dummy_y]))
            
        generator_loss.append(generator_model.train_on_batch(X_train[0:BATCH_SIZE], positive_y))
        
    # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
    generate_images(generator, args.output_dir, epoch)
    print("generator_loss="+str(generator_loss[-1]))
    print("discriminator_loss="+str(discriminator_loss[-1]))
    print("time="+str(time.time()-cur_time))
