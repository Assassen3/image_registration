import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def msi_proc(batch_path: str):
    p_folders = [f for f in os.listdir(batch_path) if
                 os.path.isdir(os.path.join(batch_path, f)) and f.isdigit()]

    for p_folder in tqdm(p_folders, desc='p文件夹', leave=False):
        p_path = os.path.join(batch_path, p_folder)

        tif_file = os.path.join(p_path, f'{p_folder}.tiff')

        try:
            msi_folder = os.path.join(p_path, 'msi_DN')
            if not os.path.exists(msi_folder):
                os.makedirs(msi_folder)

            frame = np.array(Image.open(tif_file))

            bands_order = np.array([
                21, 22, 23, 24, 25,
                16, 17, 18, 19, 20,
                11, 12, 13, 14, 15,
                6, 7, 8, 9, 10,
                1, 2, 3, 4, 5
            ])

            frame = frame[:1085, :2045]
            frame = frame.reshape(217, 5, 409, 5)
            frame = frame.transpose(0, 2, 1, 3).reshape(217, 409, 25)

            # 保存每个波段
            for i in range(25):
                current_part = frame[:, :, i]
                output_filename = os.path.join(msi_folder, f'part{bands_order[i]}.png')
                Image.fromarray(current_part).save(output_filename)

        except Exception as e:
            print(e)
    print("全部处理完毕！")


if __name__ == '__main__':
    msi_proc(r'F:\共享盘\20250510\test5\多光谱三维重构\A15')
