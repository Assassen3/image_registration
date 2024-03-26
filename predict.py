import load_data
from data_generator import data_generator_predict
from model import DNet
from losses import *

data, filenames = load_data.load_data()

predict_generator = data_generator_predict(data)

dnet = DNet()
losses = [StructuralSimilarityLoss(), DiffusionRegularLoss(), DepthDeformationLoss()]
loss_weights = [1, 0.1, 1]

dnet.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
dnet.load_weights('./weights/final/weights')

i, _ = next(predict_generator)
moved, flow, _ = dnet.predict(i)
for i in range(len(moved)):
    img = moved[i].squeeze() * 255
    img = img.astype('uint8')
    cv2.imwrite('./results/predict/' + filenames[i] + 'moved.png', img)

dnet.summary()
