import numpy as np


def data_generator(data, batch_size=10):
    vol_shape = data.shape[:-2]  # extract data shape
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        idx = np.random.randint(0, data.shape[0], size=batch_size)
        moving_images = data[idx, 0, ..., np.newaxis]
        fixed_images = data[idx, 1, ..., np.newaxis]
        depth_images = data[idx, 2, ..., np.newaxis]

        inputs = [moving_images, fixed_images, depth_images]
        outputs = [fixed_images, zero_phi, depth_images]

        yield inputs, outputs


def data_generator_predict(data, batch_size=10):
    vol_shape = data.shape[:-2]  # extract data shape
    ndims = len(vol_shape)

    zero_phi = np.zeros([data.shape[0], *vol_shape, ndims])

    idx = 0
    while True:
        if idx >= data.shape[0]:
            yield None, None
        moving_images = data[idx:idx + batch_size, -3, ..., np.newaxis]
        fixed_images = data[idx:idx + batch_size, -2, ..., np.newaxis]
        depth_images = data[idx:idx + batch_size, -1, ..., np.newaxis]
        idx += batch_size

        inputs = [moving_images, fixed_images, depth_images]
        outputs = [fixed_images, zero_phi, depth_images]

        yield inputs, outputs
