import load_data
from data_generator import data_generator
from model import DNet
from losses import *
from callbacks import DCallback
from visualize import plot_history

data, filenames = load_data.load_data()
train_generator = data_generator(data, batch_size=10)

dnet = DNet()
losses = [StructuralSimilarityLoss(), DiffusionRegularLoss(), DepthDeformationLoss()]
loss_weights = [1, 0.3, 2.5]

dnet.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

nb_epochs = 30
steps_per_epoch = 200
checkpoint_path = "./weights/{epoch:02d}-{loss:.2f}/weights"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
hist = dnet.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
                callbacks=[DCallback(), cp_callback])
dnet.summary()
dnet.save_weights('./weights/final/weights')

plot_history(hist)
