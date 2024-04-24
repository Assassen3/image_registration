import load_data
from callbacks import DCallback
from data_generator import data_generator
from losses import *
from metric import *
from model import DNet
from visualize import plot_history

data, filenames = load_data.load_data()
train_generator = data_generator(data, batch_size=10)

dnet = DNet()
losses = [StructuralSimilarityLoss(), DiffusionRegularLoss(), DepthDeformationLoss()]
loss_weights = [1, 0.3, 5]

dnet.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights,
             metrics=[[normalized_mutual_information, correlation_coefficient, calculate_ssim], [], []])

nb_epochs = 30
steps_per_epoch = 200
checkpoint_path = "./weights/20240424/{epoch:02d}-{loss:.2f}/weights"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
hist = dnet.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
                callbacks=[DCallback(), cp_callback])
dnet.save_weights('./weights/final/weights')

plot_history(hist)
