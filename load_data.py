import os

import numpy as np
from skimage import io
from tqdm import tqdm


def load_data(load_num=0):
    print('Loading data...')
    data_path = "data/tomato_modified"
    filenames = [f[:-6] for f in os.listdir(data_path) if f.endswith("ms.png")]
    load_num = len(filenames) if load_num == 0 else load_num
    load_bar = tqdm(total=load_num, unit='image')
    shape = io.imread(os.path.join(data_path, filenames[0] + 'ms.png'), as_gray=True).shape
    data = np.zeros((load_num, 3, *shape))
    data_file_name = []
    load_index = np.random.choice(len(filenames), size=load_num, replace=False)
    for i, num in enumerate(load_index):
        filename = filenames[num]
        img_ms = io.imread(data_path + "/" + filename + 'ms.png', as_gray=True)
        img_rgb = io.imread(data_path + "/" + filename + "rgb.png", as_gray=True)
        img_d = io.imread(data_path + "/" + filename + "depth.png", as_gray=True)
        img_ms = np.array(img_ms)
        img_rgb = np.array(img_rgb)
        img_d = np.array(img_d)
        # img_rgb = img_rgb.astype('float') / 255
        img_ms = img_ms.astype('float') / 255
        img_d = img_d.astype('float') / 255
        data[i, 0] = img_ms
        data[i, 1] = img_rgb
        data[i, 2] = img_d
        load_bar.update(1)
        load_bar.set_postfix({"Image": filename})
        data_file_name.append(filename)

    load_bar.close()
    return data, data_file_name
