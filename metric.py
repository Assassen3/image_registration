import tensorflow as tf


def normalized_mutual_information(y_true, y_pred):
    nbins = 200
    histy_bins = tf.histogram_fixed_width_bins(y_pred, [0., 1.], nbins=nbins)

    hist_2d = tf.map_fn(lambda i: tf.histogram_fixed_width(y_true[histy_bins == i], [0., 1.], nbins=nbins),
                        tf.range(nbins))

    pxy = hist_2d / tf.reduce_sum(hist_2d)
    px = tf.reduce_sum(pxy, axis=1)
    py = tf.reduce_sum(pxy, axis=0)

    # 计算互信息
    px_py = px[:, None] * py[None, :]
    nmi = tf.reduce_sum(pxy * tf.math.log(pxy / (px_py + 1e-10) + 1e-10))

    # 归一化
    h_x = -tf.reduce_sum(px * tf.math.log(px + 1e-10))
    h_y = -tf.reduce_sum(py * tf.math.log(py + 1e-10))
    nmi /= tf.sqrt(h_x * h_y)

    return nmi


def correlation_coefficient(y_true, y_pred):
    # 中心化
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    # 计算协方差和各自的方差
    covariance = tf.reduce_sum(y_true_centered * y_pred_centered)
    y_true_var = tf.reduce_sum(tf.square(y_true_centered))
    y_pred_var = tf.reduce_sum(tf.square(y_pred_centered))

    # 计算相关系数
    correlation = covariance / tf.sqrt(y_true_var * y_pred_var)
    return correlation


def calculate_ssim(y_pred, y_true):
    ssim = tf.image.ssim(y_pred, y_true, max_val=1)
    return tf.reduce_mean(ssim)


def mean_pred(y_pred, y_true):
    print(y_true.shape)
    return tf.reduce_mean(y_pred)
