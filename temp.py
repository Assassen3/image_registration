import cv2
from PIL import Image
import numpy as np
pts1 = np.float32([[828,319 ],
                   [910, 668 ],
                   [603, 359 ]])
pts2 = np.float32([[175, 66],
                   [199, 174],
                   [101, 72]])
M = cv2.getAffineTransform(pts1, pts2)

img= Image.open("F:/共享盘/20250521/test2/多光谱三维重构/A01/1/1_color_uint8.png")
img_cv = np.array(img)
img_cv = cv2.warpAffine(img_cv, M, (409, 217))
img_2 = Image.fromarray(img_cv)
img.show("A")
img_2.show("B")