import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import neurite as ne
import voxelmorph as vxm

# Input shapes.
in_shape = (256,) * 2
num_dim = len(in_shape)
num_label = 16
num_maps = 40

# Shape generation.
label_maps = []
for _ in tqdm.tqdm(range(num_maps)):
    # Draw image and warp.
    im = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label),
        scales=(32, 64), max_std=1,
    )
    warp = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label, num_dim),
        scales=(16, 32, 64), max_std=16,
    )

    # Transform and create label map.
    im = vxm.utils.transform(im, warp)
    lab = tf.argmax(im, axis=-1)
    label_maps.append(np.uint8(lab))

# Visualize shapes.
num_row = 2
per_row = 10
for i in range(0, num_row * per_row, per_row):
    ne.plot.slices(label_maps[i:i + per_row], cmaps=['tab20c'])

# Training-image generation. For accurate registration, the landscape of warps
# and image contrasts will need to include the target distribution.
prop = dict(
    in_shape=in_shape,
    labels_in=range(num_label),
    warp_max=4,
    warp_blur_min=(8, 8),
    warp_blur_max=(64, 64),
)

model_gen_1 = ne.models.labels_to_image_new(**prop, id=1)
model_gen_2 = ne.models.labels_to_image_new(**prop, id=2)

# Test repeatedly on the same input.
num_gen = 10
input = np.expand_dims(label_maps[0], axis=(0, -1))
slices = [model_gen_1.predict(input, verbose=0)[0] for _ in range(num_gen)]
ne.plot.slices(slices)

# Registration model.
model_def = vxm.networks.VxmDense(
    inshape=in_shape,
    int_resolution=2,
    svf_resolution=2,
    nb_unet_features=([256] * 4, [256] * 8),
    reg_field='warp',
)

# Combined model: synthesis and registration.
ima_1, map_1 = model_gen_1.outputs
ima_2, map_2 = model_gen_2.outputs

_, warp = model_def((ima_1, ima_2))
moved = vxm.layers.SpatialTransformer(fill_value=0)((map_1, warp))

inputs = (*model_gen_1.inputs, *model_gen_2.inputs)
out = (map_2, moved)
model = tf.keras.Model(inputs, out)

# Contrast invariance: MSE loss on probability maps.
model.add_loss(vxm.losses.MSE().loss(*out) + tf.repeat(0., tf.shape(moved)[0]))
model.add_loss(vxm.losses.Grad('l2', loss_mult=0.05).loss(None, warp))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Training. Re-running the cell will continue training.
hist = model.fit(
    x=vxm.generators.synthmorph(label_maps, batch_size=1, same_subj=True, flip=True),
    epochs=3,
    steps_per_epoch=100,
)

# Visualize loss.
plt.plot(hist.epoch, hist.history['loss'], '.-')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Skip training, download model weights.
model.load_weights('./weights.h5')


# Resize and normalize test images.
def conform(x, in_shape=in_shape):
    x = np.float32(x)
    x = np.squeeze(x)
    x = ne.utils.minmax_norm(x)
    x = ne.utils.zoom(x, zoom_factor=[o / i for o, i in zip(in_shape, x.shape)])
    return x[None, ..., None]


def register(moving, fixed):
    # Conform and register.
    moving = conform(moving)
    fixed = conform(fixed)
    moved, warp = model_def.predict((moving, fixed), verbose=0)

    # Visualize.
    slices = (moving, fixed, moved, warp[..., 0])
    titles = ('Moving', 'Fixed', 'Moved', 'Warp (x-axis)')
    ne.plot.slices(slices, titles, do_colorbars=True)


# Test on MNIST.
images, digits = tf.keras.datasets.mnist.load_data()[-1]
ind = np.flatnonzero(digits == 4)
register(moving=images[ind[6]], fixed=images[ind[9]])

# Test on OASIS-1.
images = ne.py.data.load_dataset('2D-OASIS-TUTORIAL')
register(moving=images[2], fixed=images[7])
