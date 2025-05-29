import os.path

import cv2
import numpy
import numpy as np
from PIL import Image

from data_generator import data_generator_predict
from load_data import *
from losses import *
from model import DNet, SpatialTransformer

DEPTH_THRESHOLD_MIN = 0
DEPTH_THRESHOLD_MAX = 2500
TARGET_SIZE = (400, 208)
SOURCE_SIZE = (1920, 1080)
CROP_BOX = (142, 466, 662, 779)
ALIGN_MATRIX = np.array(
    [[3.27437198e-01, -8.16576004e-03, -9.35131227e+01], [2.72110507e-02, 3.03062160e-01, -5.32075790e+01]])
inverse_linear_part = np.linalg.inv(ALIGN_MATRIX[:, :2])
inverse_translation_part = -np.dot(inverse_linear_part, ALIGN_MATRIX[:, 2])
M_inv = np.hstack((inverse_linear_part, inverse_translation_part.reshape(2, 1)))


def image_registration(root_path: str, DLDisable=True):
    pos_list = os.listdir(root_path)
    all_data = []
    for pos in pos_list:
        if not os.path.isdir(os.path.join(root_path, pos)):
            continue
        pos_path = os.path.join(root_path, pos)
        ms_images = []
        for i in range(25):
            img = numpy.array(Image.open(os.path.join(pos_path, "msi_DN", f"part{str(i + 1)}.png")))
            ms_images.append(img)
        rbg_image = numpy.array(Image.open(os.path.join(pos_path, f"{pos}_color_uint8.png")).convert('L'))
        depth_image = numpy.array(Image.open(os.path.join(pos_path, f"{pos}_depth_uint16.png")))
        all_data.append([rbg_image, depth_image, ms_images])
    all_data = [preprocess(*d) for d in all_data]
    for d in range(len(all_data)):
        _ = []
        for b in range(25):
            _.append(all_data[d][2][b] / 255.0)
        all_data[d] = [all_data[d][0] / 255.0, all_data[d][1] / 255.0, _]

    data = []
    for d in all_data:
        data.append([d[2][0], d[0], d[1]])
    data = np.array(data)

    if DLDisable:
        bar = tqdm(total=len(pos_list) * 25)
        for i, pos in enumerate(pos_list):
            if not os.path.isdir(os.path.join(root_path, pos)):
                continue
            ms_imgs = postprocess(all_data[i][2])
            try:
                os.mkdir(os.path.join(root_path, pos, 'predict_DN'))
            except Exception:
                pass
            for k in range(25):
                img = Image.fromarray(ms_imgs[k])
                img.save(os.path.join(root_path, pos, 'predict_DN', f'part{k + 1}.png'))
                bar.update(1)
        return

    predict_generator = data_generator_predict(data, 100)

    dnet = DNet()
    spt = SpatialTransformer()
    dnet.compile()
    dnet.load_weights('./weights/20240813/10-0.92/weights')

    count = 0
    i, _ = next(predict_generator)
    bar = tqdm(total=len(data))
    while i is not None:
        moved, flow, _ = dnet.predict(i, verbose=0)

        for j in range(len(moved)):
            flow_img = flow[j]

            ms_images = all_data[count][2]
            ms_images = tf.expand_dims(ms_images, axis=-1)
            flow_img_tf = tf.expand_dims(flow_img, axis=0)
            flow_img_tf = tf.tile(flow_img_tf, multiples=[25, 1, 1, 1])
            ms_images_moved = spt([ms_images, flow_img_tf])

            ms_images_moved = ms_images_moved.numpy().squeeze() * 256
            ms_images_moved = ms_images_moved.clip(0, 255).astype(np.uint8)

            ms_images_moved = postprocess(ms_images_moved)
            try:
                os.mkdir(os.path.join(root_path, pos_list[count], 'predict_DN'))
            except Exception:
                pass
            for k in range(25):
                img = Image.fromarray(ms_images_moved[k])
                img.save(os.path.join(root_path, pos_list[count], 'predict_DN', f'part{k + 1}.png'))
            count += 1
            bar.update(1)
        i, _ = next(predict_generator)


def preprocess(rgb_img, depth_img, ms_imgs):
    for i in range(25):
        ms_imgs[i] = numpy.array(Image.fromarray(ms_imgs[i]).resize(TARGET_SIZE))

    rgb_img = cv2.warpAffine(rgb_img, ALIGN_MATRIX, TARGET_SIZE)
    depth_img = cv2.warpAffine(depth_img, ALIGN_MATRIX, (TARGET_SIZE[0], TARGET_SIZE[1]))
    depth_img = np.where(
        (depth_img < DEPTH_THRESHOLD_MIN) | (depth_img > DEPTH_THRESHOLD_MAX),
        0,
        depth_img
    )
    depth_img = ((depth_img / DEPTH_THRESHOLD_MAX) * 255.0).astype(np.uint8)
    return rgb_img, depth_img, ms_imgs


def postprocess(ms_imgs):
    r = []
    for i in range(25):
        img = (cv2.warpAffine(ms_imgs[i], M_inv, SOURCE_SIZE) * 255).astype(np.uint8)
        r.append(img)
    return r


if __name__ == '__main__':
    image_registration(r'F:\共享盘\20250521\test2\多光谱三维重构\A01')
