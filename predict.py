import cv2
from tqdm import tqdm

from data_generator import data_generator_predict
from load_data import *
from losses import *
from model import DNet, SpatialTransformer

data, filenames = load_data("data/tomato_dn_modified")

predict_generator = data_generator_predict(data, 100)

dnet = DNet()
spt = SpatialTransformer()

dnet.compile()
dnet.load_weights('./weights/20240813/10-0.92/weights')

count = 0
i, _ = next(predict_generator)
bar = tqdm(total=len(filenames))
while i is not None:
    moved, flow, _ = dnet.predict(i, verbose=0)

    for j in range(len(moved)):
        img = moved[j]
        flow_img = flow[j]

        ms_images = load_ms_by_name("data/tomato_dn_modified", filenames[count])
        ms_images = tf.expand_dims(ms_images, axis=-1)
        flow_img_tf = tf.expand_dims(flow_img, axis=0)
        flow_img_tf = tf.tile(flow_img_tf, multiples=[25, 1, 1, 1])
        ms_images_moved = spt([ms_images, flow_img_tf])

        img = img.squeeze() * 256
        img = img.clip(0, 255).astype(np.uint8)
        ms_images_moved = ms_images_moved.numpy().squeeze() * 256
        ms_images_moved = ms_images_moved.clip(0, 255).astype(np.uint8)

        cv2.imwrite('./results/predict_dn/' + filenames[count] + 'moved_0.png', img)
        for k in range(25):
            cv2.imwrite('./results/predict_dn/' + filenames[count] + 'moved_' + str(k + 1) + '.png', ms_images_moved[k])

        min_val = np.min(flow_img)
        max_val = np.max(flow_img)
        flow_img = (flow_img - min_val) / (max_val - min_val) * 255
        flow_img = flow_img.clip(0, 255).astype(np.uint8)
        cv2.imwrite('./results/predict_dn/' + filenames[count] + 'flow_h.png', flow_img[..., 0])
        cv2.imwrite('./results/predict_dn/' + filenames[count] + 'flow_v.png', flow_img[..., 1])
        count += 1
        bar.update(1)
    i, _ = next(predict_generator)
