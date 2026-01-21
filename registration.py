import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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

cam2world = np.array([[0.978813707829, -0.151365548372, -0.137884676456, 0.144190746048],
                      [0.010250490159, -0.636350452900, 0.771331965923, -0.468420714817],
                      [-0.204496055841, -0.756403684616, -0.621316969395, 0.358610486366],
                      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])

bands_order = np.array([
    21, 22, 23, 24, 25,
    16, 17, 18, 19, 20,
    11, 12, 13, 14, 15,
    6, 7, 8, 9, 10,
    1, 2, 3, 4, 5
])


def T2T(path: Path):
    view_csv = list(path.glob('*_v*.xlsx'))[0]
    df = pd.read_excel(view_csv)  # 读取xlsx
    matrices = []

    for _, row in df.iterrows():
        degree = - int(row['DegreeRotation'])
        theta_z = np.deg2rad(degree)

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0, 0],
            [np.sin(theta_z), np.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        matrices.append(cam2world @ Rz)

    with open(path / 'T.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for mat in matrices:
            writer.writerow([mat[i // 4, i % 4] for i in range(16)])


def image_registration(root_path: Path):
    poses = list(root_path.glob("*[0-9]"))
    all_data = []

    for pos in poses:
        rbg_image = np.array(Image.open(pos / f'{pos.name}_color_uint8.png').convert('F')) / 255.0
        depth_image = np.array(Image.open(pos / f'{pos.name}_depth_uint16.png'))
        ms_images = [np.array(Image.open(pos / 'msi_DN' / f'part{i + 1}.png').convert('F')) / 255.0 for i in range(25)]
        all_data.append([rbg_image, depth_image, ms_images])

    all_data = [preprocess(*d) for d in all_data]

    for idx, pos in enumerate(tqdm(poses, leave=False)):
        ms_imgs = postprocess(all_data[idx][2])
        prd_folder = pos / 'predict_DN'
        prd_folder.mkdir(exist_ok=True)
        for band in range(25):
            img = Image.fromarray(ms_imgs[band])
            img.save(prd_folder / f'part{band + 1}.png')


def split_ms_images(batch_path: Path):
    ms_images = list(batch_path.rglob('*[0-9]/*[0-9].tiff'))

    for ms_image in tqdm(ms_images, desc='split MS images', leave=False):
        dst_folder = ms_image.parent / 'msi_DN'
        dst_folder.mkdir(parents=True, exist_ok=True)

        frame = np.array(Image.open(ms_image))

        frame = frame[:1085, :2045]
        frame = frame.reshape(217, 5, 409, 5)
        frame = frame.transpose(0, 2, 1, 3).reshape(217, 409, 25)

        for i in range(25):
            current_part = frame[:, :, i]
            output_filename = dst_folder / f'part{bands_order[i]}.png'
            Image.fromarray(current_part).save(output_filename)


def preprocess(rgb_img, depth_img, ms_imgs):
    for i in range(25):
        ms_imgs[i] = np.array(Image.fromarray(ms_imgs[i]).resize(TARGET_SIZE))

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
