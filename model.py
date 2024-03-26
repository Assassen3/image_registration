import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.conv1 = self.conv_block(64, 3, strides=1)
        self.conv2 = self.conv_block(128, 3, strides=2)
        self.conv3 = self.conv_block(256, 3, strides=2)
        self.conv4 = self.conv_block(512, 3, strides=2)

        # Bridge
        self.conv5 = self.conv_block(1024, 3, strides=2)

        # Decoder (Expansive Path)
        self.up6 = self.upconv_block(512, 2, strides=2)
        self.conv6 = self.conv_block(512, 3, strides=1)
        self.up7 = self.upconv_block(256, 2, strides=2)
        self.conv7 = self.conv_block(256, 3, strides=1)
        self.up8 = self.upconv_block(128, 2, strides=2)
        self.conv8 = self.conv_block(128, 3, strides=1)
        self.up9 = self.upconv_block(64, 2, strides=2)
        self.conv9 = self.conv_block(64, 3, strides=1)

        # Final layer
        self.conv10 = tf.keras.layers.Conv2D(2, 1)

    def conv_block(self, filters, kernel_size, strides):
        block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        return block

    def upconv_block(self, filters, kernel_size, strides):
        block = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        return block

    def call(self, inputs, **kwargs):
        # Encoder
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # Bridge
        conv5 = self.conv5(conv4)

        # Decoder
        up6 = self.up6(conv5)
        merge6 = tf.concat([conv4, up6], axis=-1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = tf.concat([conv3, up7], axis=-1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = tf.concat([conv2, up8], axis=-1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = tf.concat([conv1, up9], axis=-1)
        conv9 = self.conv9(merge9)

        # Final layer
        output = self.conv10(conv9)
        return output


class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialTransformer, self).__init__(trainable=False)

    def call(self, inputs, **kwargs):
        src = inputs[0]
        flow = inputs[1]
        shape = tf.shape(src)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # Create grid
        x_coords, y_coords = tf.meshgrid(tf.range(width), tf.range(height))
        grid = tf.stack([x_coords, y_coords], axis=-1)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.tile(grid, [batch_size, 1, 1, 1])
        grid = tf.cast(grid, dtype=tf.float32)

        # Add flow to grid
        grid += flow

        # Compute interpolation weights
        x0 = tf.floor(grid[..., 0])
        x1 = x0 + 1
        y0 = tf.floor(grid[..., 1])
        y1 = y0 + 1

        wa = tf.expand_dims((x1 - grid[..., 0]) * (y1 - grid[..., 1]), axis=-1)
        wb = tf.expand_dims((grid[..., 0] - x0) * (y1 - grid[..., 1]), axis=-1)
        wc = tf.expand_dims((x1 - grid[..., 0]) * (grid[..., 1] - y0), axis=-1)
        wd = tf.expand_dims((grid[..., 0] - x0) * (grid[..., 1] - y0), axis=-1)

        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        x0 = tf.expand_dims(x0, axis=-1)
        x1 = tf.expand_dims(x1, axis=-1)
        y0 = tf.expand_dims(y0, axis=-1)
        y1 = tf.expand_dims(y1, axis=-1)

        batch_indices = tf.range(batch_size)
        batch_indices = tf.expand_dims(batch_indices, axis=-1)
        batch_indices = tf.expand_dims(batch_indices, axis=-1)
        batch_indices = tf.tile(batch_indices, [1, height, width])
        batch_indices = tf.expand_dims(batch_indices, axis=-1)

        last_indices = tf.zeros([batch_size, height, width, 1], dtype=tf.int32)

        indices_a = tf.concat([batch_indices, y0, x0, last_indices], axis=-1)
        indices_b = tf.concat([batch_indices, y0, x1, last_indices], axis=-1)
        indices_c = tf.concat([batch_indices, y1, x0, last_indices], axis=-1)
        indices_d = tf.concat([batch_indices, y1, x1, last_indices], axis=-1)

        pixel_values_a = tf.expand_dims(tf.gather_nd(src, indices_a), axis=-1)
        pixel_values_b = tf.expand_dims(tf.gather_nd(src, indices_b), axis=-1)
        pixel_values_c = tf.expand_dims(tf.gather_nd(src, indices_c), axis=-1)
        pixel_values_d = tf.expand_dims(tf.gather_nd(src, indices_d), axis=-1)

        warped_image = wa * pixel_values_a + wb * pixel_values_b + wc * pixel_values_c + wd * pixel_values_d

        return warped_image


class DNet(tf.keras.Model):
    def __init__(self):
        super(DNet, self).__init__()
        self.unet = UNet()
        self.stn = SpatialTransformer()

    def call(self, inputs, **kwargs):
        moving = inputs[0]
        fixed = inputs[1]
        depth = inputs[2]
        flow = self.unet(tf.concat([moving, fixed, depth], axis=3))
        moved_image = self.stn.call([moving, flow])
        return moved_image, flow, flow
