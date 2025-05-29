import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from all_togather import main, run_cpp_program
# import metashape
# import os
# # import argparse
# import matlab
# import subprocess

# print("Current working directory:", os.getcwd())
# 计算并显示平均图像
def ThreeD_PointCloud():
    folder_path = folder_path_label.cget("text")
    # 遍历所有子文件夹
    # 这个方法会遍历 parent_folder 下的所有目录，返回root,当前遍历到的目录路径。dirs：子文件夹名称列表list，files：当前目录下的所有文件名称列表（不包括子目录）。
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs: #遍历子目录中的文件,folder为子文件夹的名字
            sub_folder_path = os.path.join(root, folder)#构造子文件夹的完整路径
            print(sub_folder_path)
            os.system('D:\soft\metashape2.1.3\\metashape -r D:\image_registraion_using_depth_by_Fang\\Bulid_3D.py' + " " + sub_folder_path + " " + folder + " " + root)

def ThreeD_MPC():
    print("Running MATLAB program...")
    folder_path1 = folder_path_label1.cget("text")
    # 遍历所有子文件夹
    # 这个方法会遍历 parent_folder 下的所有目录，返回root,当前遍历到的目录路径。dirs：子文件夹名称列表list，files：当前目录下的所有文件名称列表（不包括子目录）。
    for root, dirs, files in os.walk(folder_path1):
        for folder in dirs: #遍历子目录中的文件,folder为子文件夹的名字
            sub_folder_path = os.path.join(root, folder)#构造子文件夹的完整路径
            print(sub_folder_path)
            main(sub_folder_path)
    
# 选择路径
#三维重建路径
def select_folder():
    folder_path = filedialog.askdirectory()
    folder_path_label.config(text=folder_path)

#三维多光谱点云路径
def select_folder1():
    folder_path1 = filedialog.askdirectory()
    folder_path_label1.config(text=folder_path1)

# 创建主窗口
root = tk.Tk()
root.title("Dataprocess")

# 设置窗口初始大小
root.geometry("600x400")

# 创建按钮和标签
#颜色校正路径选择按钮创建
select_folder_button = tk.Button(root, text="选择文件夹", command=select_folder)
folder_path_label = tk.Label(root, text="未选择文件夹", width=40, anchor="w")

#三维光谱点云路径选择按钮创建
select_folder_button1 = tk.Button(root, text="选择文件夹", command=select_folder1)
folder_path_label1 = tk.Label(root, text="未选择文件夹", width=40, anchor="w")

#三维多视角重建按钮创建
ThreeD_PointCloud_button = tk.Button(root, text="ThreeD_PointCloud", command=ThreeD_PointCloud)

#三维多光谱点云重建按钮创建
ThreeD_MPC_button = tk.Button(root, text="ThreeD_MPC", command=ThreeD_MPC)

# result_label = tk.Label(root, text="结果：")

# 创建标签来显示结果图片
# result_image_label = tk.Label(root)

# 布局
select_folder_button.grid(row=0, column=0, padx=10, pady=10)
folder_path_label.grid(row=0, column=1, padx=10, pady=10)
ThreeD_PointCloud_button.grid(row=1, column=0, padx=10, pady=10)
select_folder_button1.grid(row=2, column=0, padx=10, pady=10)
folder_path_label1.grid(row=2, column=1, padx=10, pady=10)
ThreeD_MPC_button.grid(row=3, column=0, padx=10, pady=10)

# result_label.grid(row=1, column=1, padx=10, pady=10)
# result_image_label.grid(row=2, column=1, padx=10, pady=10)

# 运行主循环
root.mainloop()
