import os

import matplotlib.pyplot as plt
import neurite as ne
import numpy as np
import voxelmorph as vxm
from skimage import io
from tqdm import tqdm

from data_generator import data_generator

data_path = "data/tomato_modified"
filenames = [f for f in os.listdir(data_path) if f.endswith("ms.png")]
load_bar = tqdm(total=len(filenames), unit='image')
data = np.zeros((len(filenames), 2, 192, 384))
for num, filename in enumerate(filenames):
    img_ms = io.imread(data_path + "/" + filename, as_gray=True)
    img_rgb = io.imread(data_path + "/" + filename[:-6] + "rgb.png", as_gray=True)
    img_ms = np.array(img_ms)
    img_rgb = np.array(img_rgb)
    img_ms = img_ms.astype('float') / 255
    data[num, 0] = img_ms
    data[num, 1] = img_rgb
    load_bar.update(1)
    load_bar.set_postfix({"Image": filename})

load_bar.close()

nb_features = [
    [32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]
# build model using VxmDense
inshape = data.shape[2:]

train_generator = data_generator(data)

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# let's test it

in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

nb_epochs = 10
steps_per_epoch = 100
hist = vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


plot_history(hist)

# let's get some data
val_generator = data_generator(data, batch_size=1)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
# visualize
images = [img[0, :, :, 0] for img in val_input + val_pred]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
ne.plot.flow([val_pred[1].squeeze()], width=5)
