import tkinter as tk
from tkinter import filedialog
import os
import threading
from all_togather import main

# 计算并显示平均图像的线程函数
def ThreeD_PointCloud_thread():
    folder_path = folder_path_label.cget("text")
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs:
            sub_folder_path = os.path.join(root, folder)
            print(sub_folder_path)
            os.system('D:\soft\metashape2.1.3\\metashape -r D:\image_registraion_using_depth_by_Fang\\Bulid_3D.py' + " " + sub_folder_path + " " + folder + " " + root)

# 三维多光谱点云的线程函数
def ThreeD_MPC_thread():
    print("Running MATLAB program...")
    folder_path1 = folder_path_label1.cget("text")
    for root, dirs, files in os.walk(folder_path1):
        for folder in dirs:
            sub_folder_path = os.path.join(root, folder)
            print(sub_folder_path)
            main(sub_folder_path)

# 启动 ThreeD_PointCloud 的线程
def ThreeD_PointCloud():
    thread = threading.Thread(target=ThreeD_PointCloud_thread)
    thread.start()

# 启动 ThreeD_MPC 的线程
def ThreeD_MPC():
    thread = threading.Thread(target=ThreeD_MPC_thread)
    thread.start()

# 选择路径 - 三维重建路径
def select_folder():
    folder_path = filedialog.askdirectory()
    folder_path_label.config(text=folder_path)

# 选择路径 - 三维多光谱点云路径
def select_folder1():
    folder_path1 = filedialog.askdirectory()
    folder_path_label1.config(text=folder_path1)

# 创建主窗口
root = tk.Tk()
root.title("Dataprocess")

# 设置窗口初始大小
root.geometry("600x400")

# 创建按钮和标签
select_folder_button = tk.Button(root, text="选择文件夹", command=select_folder)
folder_path_label = tk.Label(root, text="未选择文件夹", width=40, anchor="w")

select_folder_button1 = tk.Button(root, text="选择文件夹", command=select_folder1)
folder_path_label1 = tk.Label(root, text="未选择文件夹", width=40, anchor="w")

ThreeD_PointCloud_button = tk.Button(root, text="ThreeD_PointCloud", command=ThreeD_PointCloud)
ThreeD_MPC_button = tk.Button(root, text="ThreeD_MPC", command=ThreeD_MPC)

# 布局
select_folder_button.grid(row=0, column=0, padx=10, pady=10)
folder_path_label.grid(row=0, column=1, padx=10, pady=10)
ThreeD_PointCloud_button.grid(row=1, column=0, padx=10, pady=10)
select_folder_button1.grid(row=2, column=0, padx=10, pady=10)
folder_path_label1.grid(row=2, column=1, padx=10, pady=10)
ThreeD_MPC_button.grid(row=3, column=0, padx=10, pady=10)

# 运行主循环
root.mainloop()
