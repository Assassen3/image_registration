import cv2
import numpy as np

from losses import *
import tensorflow as tf
a = cv2.imread('results/20230315_p1_3_rgb.png', cv2.IMREAD_GRAYSCALE)
b = cv2.imread('results/20230315_p1_3_ms.png', cv2.IMREAD_GRAYSCALE)
a = a.astype(np.float32) / 255.0
b = b.astype(np.float32) / 255.0
a = tf.constant(a)
b = tf.constant(b)
a = a[tf.newaxis, ..., tf.newaxis]
b = b[tf.newaxis, ..., tf.newaxis]
c = tf.concat([a, b], axis=-1)
loss1 = StructuralSimilarityLoss()
loss2 = DiffusionRegularLoss()
loss3 = DepthDeformationLoss()
a1 = loss1.call(a, b)
b1 = loss2.call(a, c)
c1 = loss3.call(a, c)
print(1)
