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

        loss = (tf.reduce_mean(tf.square(dy), axis=[1, 2, 3]) + tf.reduce_mean(tf.square(dx), axis=[1, 2, 3])
                + tf.reduce_mean(tf.square(dy2), axis=[1, 2, 3]) + tf.reduce_mean(tf.square(dyx), axis=[1, 2, 3])
                + tf.reduce_mean(tf.square(dxy), axis=[1, 2, 3]) + tf.reduce_mean(tf.square(dx2), axis=[1, 2, 3]))

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
        valid_depth_mask = tf.cast(tf.not_equal(depth_map, 0.0), tf.float32)

        # 计算深度图的梯度
        depth_grad_y, depth_grad_x = tf.image.image_gradients(depth_map)

        # 计算变形场的梯度
        def_grad_yy, def_grad_yx = tf.image.image_gradients(deformation_field[..., 0, tf.newaxis])
        def_grad_xy, def_grad_xx = tf.image.image_gradients(deformation_field[..., 1, tf.newaxis])

        # 计算深度图梯度和变形场梯度之间的差异
        diff_y = tf.square(depth_grad_y - def_grad_yy) + tf.square(depth_grad_y - def_grad_xy)
        diff_x = tf.square(depth_grad_x - def_grad_yx) + tf.square(depth_grad_x - def_grad_xx)

        # 应用有效深度值掩码
        masked_diff_y = diff_y * valid_depth_mask
        masked_diff_x = diff_x * valid_depth_mask

        # 计算损失
        loss = tf.reduce_mean(masked_diff_y, axis=[1, 2, 3]) + tf.reduce_mean(masked_diff_x, axis=[1, 2, 3])

        return loss
