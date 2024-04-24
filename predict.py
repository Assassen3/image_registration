import load_data
from data_generator import data_generator_predict
from losses import *
from metric import SSIM
from model import DNet

data, filenames = load_data.load_data()

predict_generator = data_generator_predict(data, 100)

dnet = DNet()
losses = [StructuralSimilarityLoss(), DiffusionRegularLoss(), DepthDeformationLoss()]
loss_weights = [1, 0.3, 2.5]
ssim_metric = SSIM()

dnet.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights, metrics=ssim_metric)
dnet.load_weights('./weights/final/weights')

count = 0
i, _ = next(predict_generator)
while i is not None:
    moved, flow, _ = dnet.predict(i, verbose=0)
    for img in moved:
        img = img.squeeze() * 256
        img = img.clip(0, 255).astype(np.uint8)
        cv2.imwrite('./results/predict/' + filenames[count] + 'moved.png', img)
        count += 1
    i, _ = next(predict_generator)
