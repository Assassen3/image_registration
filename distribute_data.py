import os
import shutil

from tqdm import tqdm


def copy_folder_structure(src, dst):
    for root, dirs, files in os.walk(src):
        # Construct the destination path
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        # Create the directory at the destination path
        os.makedirs(dst_path, exist_ok=True)

        # Optionally, print the created directories
        # print(f"Created directory: {dst_path}")


# 源文件夹路径
src_folder = 'D:\\files\\毕业设计\\tomato_data_organs_pose'
# 目标文件夹路径
dst_folder = 'E:\\tomato_data_organs_pose_registration\\'

copy_folder_structure(src_folder, dst_folder)
print('folder structure copied successfully!')

root_path = "D:\\files\\ComputerScience\\Programs\\image_registration\\data\\tomato_dn_back_modified"
dst_path = "E:\\tomato_data_organs_pose_registration\\"
filenames = os.listdir(root_path)
bar = tqdm(filenames)

for file in filenames:
    date, p, n, ms, part = file.split("_")
    dir_path = dst_path + date + "\\" + p + "\\" + n + "\\" + "msi_moved"
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    shutil.copy(os.path.join(root_path, file), os.path.join(dir_path, "part" + part))
    bar.update(1)
