import matplotlib.pyplot as plt
import numpy as np

from model import SpatialTransformer

spt = SpatialTransformer()
img = np.zeros((1, 255, 255, 1), dtype=np.float32)
for i in range(255):
    img[0, i, :, 0] = i
flow = np.zeros((1, 255, 255, 2), dtype=np.float32)
for i in range(50, 100):
    flow[0, i, :, 1] = 25.

img_moved = spt([img, flow])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(img[0, ..., 0])
ax2.imshow(flow[0, ..., 1])
ax3.imshow(img_moved[0, ..., 0])
plt.tight_layout()
plt.show()
