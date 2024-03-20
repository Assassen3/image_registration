import matplotlib.pyplot as plt
import neurite as ne
import voxelmorph as vxm
import load_data
from data_generator import data_generator
from model import DNet
from losses import *

data, filenames = load_data.load_data()

inshape = data.shape[2:]

train_generator = data_generator(data, batch_size=10)

dnet = DNet()
losses = [StructuralSimilarityLoss(), DiffusionRegularLoss(), DepthDeformationLoss()]

# usually, we have to balance the two losses by a hyper-parameter
loss_weights = [1, 0.0005, 0.000001]

dnet.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
# in_sample, out_sample = next(train_generator)
#
# # visualize
# images = [img[0, :, :, 0] for img in in_sample + out_sample]
# titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# i, o = next(train_generator)
# p = dnet(i)
# a = StructuralSimilarityLoss()
# b = DiffusionRegularLoss()
# c = DepthDeformationLoss()
# a1 = a.call(o[0], p[0])
# b1 = b.call(o[1], p[1])
# c1 = c.call(o[2], p[2])

nb_epochs = 10
steps_per_epoch = 200
hist = dnet.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)
dnet.save_weights('./weights/dnet_1.h5')


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


plot_history(hist)
