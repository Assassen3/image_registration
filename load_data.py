from tqdm import tqdm
import numpy as np
from skimage import io
import os


def load_data(load_num=0):
    print('Loading data...')
    data_path = "data/tomato_modified2"
    filenames = [f for f in os.listdir(data_path) if f.endswith("ms.png")]
    load_num = len(filenames) if load_num == 0 else load_num
    load_bar = tqdm(total=load_num, unit='image')
    # data = np.zeros((load_num, 2, 192, 384))
    data = np.zeros((load_num, 2, 256, 256))
    for num, filename in enumerate(filenames[:load_num]):
        img_ms = io.imread(data_path + "/" + filename, as_gray=True)
        img_rgb = io.imread(data_path + "/" + filename[:-6] + "rgb.png", as_gray=True)
        img_ms = np.array(img_ms)
        img_rgb = np.array(img_rgb)
        img_ms = img_ms.astype('float') / 255
        data[num, 0] = img_ms
        data[num, 1] = img_rgb
        load_bar.update(1)
        load_bar.set_postfix({"Image": filename})
        if num == load_num:
            break
    load_bar.close()
    return data
