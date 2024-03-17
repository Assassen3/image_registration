import os
import matplotlib.pyplot as plt
import neurite as ne
import numpy as np
import voxelmorph as vxm
from skimage import io
from tqdm import tqdm
import cv2

import load_data
from data_generator import data_generator_predict

data, _ = load_data.load_data(20)

# nb_features = [
#     [64, 64, 64, 64],  # encoder features
#     [64, 64, 64, 64, 64, 32]  # decoder features
# ]
nb_features = [
    [128, 128, 256, 256, 256],  # encoder features
    [256, 256, 128, 128, 64, 32]  # decoder features
]
# build model using VxmDense
inshape = data.shape[2:]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
vxm_model.load_weights('./weights/vxm_model_2563.h5')
# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.02
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# let's test it


# let's get some data
val_generator = data_generator_predict(data)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)

bar = tqdm(total=len(val_pred[0]))
for num, _ in enumerate(val_pred[0]):
    # 获取变形后的 moving 图像和变形场
    warped_moving_image = val_pred[0][num, ..., 0]  # 变形后的 moving 图像
    displacement_field = val_pred[1][num, ...]  # 变形场

    # plt.imsave('./results/tomato4/' + str(num) + '.png', warped_moving_image, cmap='gray')
    bar.update(1)

# #显示图用
# for i in range(val_pred[0].shape[0]):
#     images = [img[i, :, :, 0] for img in val_input + val_pred]
#     titles = ['moving', 'fixed', 'moved', 'flow']
#     ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
#     flow = val_pred[1][np.newaxis, i, ...]
#     ne.plot.flow([flow.squeeze()], width=5)

