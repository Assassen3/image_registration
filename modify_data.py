import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class Modifyer:
    def __init__(self, data_list, target_size):
        self.data_list = data_list
        self.target_size = target_size

    def modify_data(self):
        assert len(self.data_list["rgb"]) == len(self.data_list["ms"]) == len(self.data_list["depth"]), "图片读入数量不一致！！"
        progress_bar_ms = tqdm(total=len(self.data_list["rgb"]), unit="image")
        datas = {
            "rgb": [],
            "ms": [],
            "depth": []
        }
        for index in range(len(self.data_list["rgb"])):
            datas["ms"].append([])
            for i in range(25):
                with Image.open(self.data_list["ms"][index][i]) as ms_img:
                    resized_img = ms_img.resize(self.target_size)
                    datas["ms"][-1].append(resized_img)

            with Image.open(self.data_list["rgb"][index]) as rgb_img:
                flipped_img = rgb_img.transpose(Image.ROTATE_270)
                crop_box = (142, 466, 662, 779)
                cropped_img = flipped_img.crop(crop_box)
                resized_img = cropped_img.resize(target_size)
                datas["rgb"].append(resized_img)

            with Image.open(self.data_list["depth"][index]) as d_img:
                flipped_img = d_img.transpose(Image.ROTATE_270)
                crop_box = (142, 466, 662, 779)
                cropped_img = flipped_img.crop(crop_box)
                resized_img = cropped_img.resize(target_size)
                depth_image = np.array(resized_img)
                # 定义异常值的阈值
                threshold_min = 0  # 最小深度值
                threshold_max = 2500  # 最大深度值

                # 识别异常值
                mask = np.logical_or(depth_image < threshold_min, depth_image > threshold_max)

                # 过滤异常值
                filtered_depth_image = np.where(mask, 0.0, depth_image)

                scaled_depth_image = (filtered_depth_image / threshold_max) * 255.0
                # 将数据类型转换为8位无符号整数
                scaled_depth_image = scaled_depth_image.astype(np.uint8)
                datas["depth"].append(scaled_depth_image)

            # 更新进度条
            progress_bar_ms.update(1)
            progress_bar_ms.set_postfix({"Image": self.data_list["rgb"][index]})

        print("图片批量处理完成!")


if __name__ == '__main__':

    # 指定输入图片所在的目录
    input_dir = ".\\data\\tomato_dn"

    # 指定输出图片的目录
    output_dir = ".\\data\\tomato_dn_modified"

    # 指定目标分辨率
    # TODO crop image instead of resize
    target_size = (400, 208)

    # 获取输入目录中的所有图片文件
    image_files_ms = [f[:-7] for f in os.listdir(input_dir) if f.endswith("rgb.png")]

    progress_bar_ms = tqdm(total=len(image_files_ms), unit="image")

    for filename in image_files_ms:
        filepath = os.path.join(input_dir, filename)

        for i in range(25):
            with Image.open(filepath + 'ms_' + str(i + 1) + '.png') as ms_img:
                resized_img = ms_img.resize(target_size)
                output_filepath = os.path.join(output_dir, filename + 'ms_' + str(i + 1) + '.png')
                resized_img.save(output_filepath)

        with Image.open(filepath + 'rgb.png') as rgb_img:
            flipped_img = rgb_img.transpose(Image.ROTATE_270)
            crop_box = (142, 466, 662, 779)
            cropped_img = flipped_img.crop(crop_box)
            resized_img = cropped_img.resize(target_size)
            output_filepath = os.path.join(output_dir, filename + 'rgb.png')
            resized_img.save(output_filepath)

        with Image.open(filepath + 'depth.png') as d_img:
            flipped_img = d_img.transpose(Image.ROTATE_270)
            crop_box = (142, 466, 662, 779)
            cropped_img = flipped_img.crop(crop_box)
            resized_img = cropped_img.resize(target_size)
            depth_image = np.array(resized_img)
            # 定义异常值的阈值
            threshold_min = 0  # 最小深度值
            threshold_max = 2500  # 最大深度值

            # 识别异常值
            mask = np.logical_or(depth_image < threshold_min, depth_image > threshold_max)

            # 过滤异常值
            filtered_depth_image = np.where(mask, 0.0, depth_image)

            scaled_depth_image = (filtered_depth_image / threshold_max) * 255.0
            # 将数据类型转换为8位无符号整数
            scaled_depth_image = scaled_depth_image.astype(np.uint8)
            output_filepath = os.path.join(output_dir, filename + 'depth.png')
            cv2.imwrite(output_filepath, scaled_depth_image)

        # 更新进度条
        progress_bar_ms.update(1)
        progress_bar_ms.set_postfix({"Image": filename})

    print("图片批量处理完成!")
