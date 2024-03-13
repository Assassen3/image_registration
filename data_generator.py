import numpy as np


def data_generator(data, batch_size=25):
    vol_shape = data.shape[2:]  # extract data shape
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        idx = np.random.randint(0, data.shape[0], size=batch_size)
        moving_images = data[idx, 0, ..., np.newaxis]
        fixed_images = data[idx, 1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        outputs = [fixed_images, zero_phi]

        yield inputs, outputs


def data_generator_predict(data):
    vol_shape = data.shape[2:]  # extract data shape
    ndims = len(vol_shape)

    zero_phi = np.zeros([data.shape[0], *vol_shape, ndims])

    while True:
        moving_images = data[:, 0, ..., np.newaxis]
        fixed_images = data[:, 1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        outputs = [fixed_images, zero_phi]

        yield inputs, outputs
