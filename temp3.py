import os
import shutil
from tqdm import tqdm

root_path = r"E:\files\ComputerScience\Programs\image_registration\results\predict"
dst_path = "G:\\"
file_list = os.listdir(root_path)
bar = tqdm(total=len(file_list))
for file in file_list:
    shutil.copy(os.path.join(root_path, file), os.path.join(dst_path, file))
    bar.update(1)
