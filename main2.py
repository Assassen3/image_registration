import matplotlib.pyplot as plt
import neurite as ne
import numpy as np
import voxelmorph as vxm
from skimage import io

moving_path = 'results/multispectral_image.jpg'
fixed_path = 'results/rgb_image.jpg'
moving_image = io.imread(moving_path, as_gray=True)
moving_image = np.array(moving_image)
fixed_image = io.imread(fixed_path, as_gray=True)
fixed_image = np.array(fixed_image)

nb_features = [
    [32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]
# build model using VxmDense
inshape = moving_image.shape


def data_generator(m, f):
    vol_shape = m.shape
    ndims = len(vol_shape)
    zero_phi = np.zeros([1, *vol_shape, ndims])
    m = m[np.newaxis, ..., np.newaxis]
    f = f[np.newaxis, ..., np.newaxis]
    inputs = [m, f]
    outputs = [f, zero_phi]
    while True:
        yield inputs, outputs


dg = data_generator(moving_image, fixed_image)

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# let's test it

in_sample, out_sample = next(dg)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

nb_epochs = 10
steps_per_epoch = 100
hist = vxm_model.fit(dg, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


plot_history(hist)

# let's get some data

val_input, _ = next(dg)
val_pred = vxm_model.predict(val_input)
# visualize
images = [img[0, :, :, 0] for img in val_input + val_pred]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
ne.plot.flow([val_pred[1].squeeze()], width=5)
