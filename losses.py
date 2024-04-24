import numpy as np
import tensorflow as tf


class StructuralSimilarityLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(StructuralSimilarityLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.constant(y_true, dtype=tf.float32) if not tf.is_tensor(y_true) else y_true
        y_pred = tf.consant(y_pred, dtype=tf.float32) if not tf.is_tensor(y_pred) else y_pred

        ssim = tf.image.ssim(y_true, y_pred, max_val=1)
        return 1 - ssim


class DiffusionRegularLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(DiffusionRegularLoss, self).__init__()

    def call(self, _, y_pred):
        return self.compute_loss(y_pred)

    def compute_loss(self, y_pred):
        dy, dx = tf.image.image_gradients(y_pred)
        dy2, dyx = tf.image.image_gradients(dy)
        dxy, dx2 = tf.image.image_gradients(dx)

        dy = tf.reduce_mean(tf.square(dy), axis=[1, 2, 3])
        dx = tf.reduce_mean(tf.square(dx), axis=[1, 2, 3])
        dy2 = tf.reduce_mean(tf.square(dy2), axis=[1, 2, 3])
        dyx = tf.reduce_mean(tf.square(dyx), axis=[1, 2, 3])
        dxy = tf.reduce_mean(tf.square(dxy), axis=[1, 2, 3])
        dx2 = tf.reduce_mean(tf.square(dx2), axis=[1, 2, 3])

        loss = dy + dx + dy2 + dyx + dxy + dx2

        return loss


class DepthDeformationLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(DepthDeformationLoss, self).__init__()

    def call(self, y_true, y_pred):
        depth_map, deformation_field = y_true, y_pred
        return self.compute_loss(depth_map, deformation_field)

    def compute_loss(self, depth_map, deformation_field):
        depth_map = tf.constant(depth_map, dtype=tf.float32) if not tf.is_tensor(depth_map) else depth_map
        deformation_field = tf.consant(deformation_field, dtype=tf.float32) \
            if not tf.is_tensor(deformation_field) else deformation_field

        # 获取有效深度值的掩码
        valid_depth_mask = tf.not_equal(depth_map, 0.0)

        loss = self.dssim(depth_map, deformation_field, valid_depth_mask)

        return loss

    def dssim(self, img1, img2, mask, k1=0.01, k2=0.03, win_size=11, L=1):
        img2 = tf.sqrt(tf.reduce_sum(tf.square(img2), axis=-1)[..., tf.newaxis])
        half_win = win_size // 2

        paddings = tf.constant([[0, 0], [half_win, half_win], [half_win, half_win], [0, 0]])
        # 对图像进行填充，使得可以对边缘像素进行处理
        pad_img1 = tf.pad(img1, paddings, mode='reflect')
        pad_img2 = tf.pad(img2, paddings, mode='reflect')
        pad_mask = tf.cast(tf.pad(mask, paddings, mode='CONSTANT'), tf.float32)
        dfilter = tf.constant(np.ones([win_size, win_size])[..., np.newaxis, np.newaxis], dtype=tf.float32)
        masked = tf.nn.conv2d(pad_mask, dfilter, strides=[1, 1, 1, 1], padding='VALID')

        mu1 = tf.nn.conv2d(pad_img1, dfilter, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(pad_img2, dfilter, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(pad_img1 ** 2, dfilter, strides=[1, 1, 1, 1], padding='VALID')
        sigma2_sq = tf.nn.conv2d(pad_img2 ** 2, dfilter, strides=[1, 1, 1, 1], padding='VALID')
        sigma12 = tf.nn.conv2d(pad_img1 * pad_img2, dfilter, strides=[1, 1, 1, 1], padding='VALID')

        # 计算SSIM图
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        masked = tf.cast(tf.less(masked, 100), tf.float32)
        # 计算平均SSIM值
        mssim = ssim_map * masked

        return mssim
