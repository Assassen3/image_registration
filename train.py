import os

import matplotlib.pyplot as plt
import neurite as ne
import numpy as np
import voxelmorph as vxm
from skimage import io
from tqdm import tqdm
import load_data

from data_generator import data_generator

data = load_data.load_data()

# nb_features = [
#     [64, 64, 64, 64],  # encoder features
#     [64, 64, 64, 64, 64, 32]  # decoder features
# ]
nb_features = [
    [128, 128, 256, 256, 256],  # encoder features
    [256, 256, 128, 128, 64, 32]  # decoder features
]
# build model using VxmDense
inshape = data.shape[2:]

train_generator = data_generator(data, batch_size=25)

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.5
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# let's test it

in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

nb_epochs = 100
steps_per_epoch = 100
hist = vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)
vxm_model.save_weights('./weights/vxm_model_2564.h5')


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


plot_history(hist)
