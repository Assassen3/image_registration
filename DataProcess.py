import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
# import metashape
import os
# import argparse



# print("Current working directory:", os.getcwd())
# 计算并显示平均图像
def ThreeD_PpintCloud():
    folder_path = folder_path_label.cget("text")
    # 遍历所有子文件夹
    # 这个方法会遍历 parent_folder 下的所有目录，返回root,当前遍历到的目录路径。dirs：子文件夹名称列表list，files：当前目录下的所有文件名称列表（不包括子目录）。
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs: #遍历子目录中的文件,folder为子文件夹的名字
            sub_folder_path = os.path.join(root, folder)#构造子文件夹的完整路径
            print(sub_folder_path)
            print(folder)
            print(root)
            os.system('D:\soft\metashape2.1.3\\metashape -r E:\\GUOQinghui\\Code\\guo\\phnotyper\\GUO\\Bulid_3D.py' + " " + sub_folder_path + " " + folder + " " + root)
            # C:\\Users\\zjdx\\Desktop\\phnotyper\\GUO



# 选择路径
def select_folder():
    folder_path = filedialog.askdirectory()
    folder_path_label.config(text=folder_path)

# 创建主窗口
root = tk.Tk()
root.title("Dataprocess")

# 设置窗口初始大小
root.geometry("600x400")

# 创建按钮和标签
select_folder_button = tk.Button(root, text="选择文件夹", command=select_folder)
folder_path_label = tk.Label(root, text="未选择文件夹", width=40, anchor="w")

add_button = tk.Button(root, text="ThreeD_PpintCloud", command=ThreeD_PpintCloud)

result_label = tk.Label(root, text="结果：")

# 创建标签来显示结果图片
result_image_label = tk.Label(root)

# 布局
select_folder_button.grid(row=0, column=0, padx=10, pady=10)
folder_path_label.grid(row=0, column=1, padx=10, pady=10)
add_button.grid(row=1, column=0, padx=10, pady=10)
result_label.grid(row=1, column=1, padx=10, pady=10)
result_image_label.grid(row=2, column=1, padx=10, pady=10)

# 运行主循环
root.mainloop()
