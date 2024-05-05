import os
import shutil
from tqdm import tqdm

root_path = "E:\\files\\ComputerScience\\Programs\\image_registration\\data\\tomato_back_modified"
dst_path = "E:\\files\\毕业设计\\tomato_data_organs_pose\\"

filenames = [f[:-9] for f in os.listdir(root_path) if f.endswith("_ms_1.png")]
bar = tqdm(filenames)

for file in filenames:
    date, p, n = file.split("_")
    dir_path = dst_path + date + "\\" + p + "\\" + n + "\\" + "moved_msi"
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        print(f"文件夹 '{dir_path}' 已存在。")
    except Exception as e:
        print(f"创建文件夹时出错：{e}")
    shutil.copy(os.path.join(root_path, file + "_ms_1.png"), os.path.join(dir_path, "1_moved_msi.png"))
    bar.update(1)

