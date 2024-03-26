import tensorflow as tf
import numpy as np
import cv2
from skimage import io


class DCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        rgb = io.imread('results/20230315_p1_3_rgb.png',as_gray=True)
        ms = io.imread('results/20230315_p1_3_ms.png', as_gray=True)
        depth = io.imread('results/20230315_p1_3_depth.png', as_gray=True)
        ms = np.array(ms)
        rgb = np.array(rgb)
        depth = np.array(depth)
        ms = ms.astype('float32') / 255.0
        depth = depth.astype('float32') / 255.0
        rgb = rgb[np.newaxis, ..., np.newaxis]
        ms = ms[np.newaxis, ..., np.newaxis]
        depth = depth[np.newaxis, ..., np.newaxis]
        moved_image, flow, _ = self.model.predict([ms, rgb, depth])
        moved_image = np.squeeze(moved_image) * 255
        moved_image = moved_image.astype('uint8')
        flow = np.squeeze(flow)
        cv2.imwrite('results/training/' + str(epoch) + '_moved_image.png', moved_image)
        # cv2.imwrite('results/training/' + str(epoch) + '_flow.png', flow)
