import os
import shutil

root_path = "E:\\files\\毕业设计\\tomato_data_organs_pose\\"
dst_path = "E:\\files\\ComputerScience\\Programs\\image_registration\\data\\tomato"
date_list = os.listdir(root_path)
for date in date_list:
    pos_list = os.listdir(root_path + date)
    for pos in pos_list:
        pos_path = os.path.join(root_path, date, pos)
        n_list = os.listdir(pos_path)
        for n in n_list:
            n_path = os.path.join(pos_path, n)
            if os.path.isdir(n_path) and n.isdigit():
                ms_image = os.path.join(n_path, "msi_reflectance\\part1.png")
                rbg_image = os.path.join(n_path, n + "_color_uint8.png")
                depth_image = os.path.join(n_path, n + "_depth_uint16.png")
                shutil.copy(ms_image, os.path.join(dst_path, date + "_" + pos + "_" + n + '_ms.png'))
                shutil.copy(rbg_image, os.path.join(dst_path, date + "_" + pos + "_" + n + '_rgb.png'))
                shutil.copy(depth_image, os.path.join(dst_path, date + "_" + pos + "_" + n + '_depth.png'))
