import os
import shutil
import cv2

# def copy_folder_structure(src, dst):
#     for root, dirs, files in os.walk(src):
#         # Construct the corresponding destination path
#         dst_root = os.path.join(dst, os.path.relpath(root, src))
#
#         # Create the directories in the destination
#         for dir in dirs:
#             os.makedirs(os.path.join(dst_root, dir), exist_ok=True)
#
#
# # 使用示例
# source_folder = r'E:\files\毕业设计\tomato_data_organs_pose'
# destination_folder = r'H:\tomato_data_organs_pose'
#
# # 确保目标文件夹存在
# os.makedirs(destination_folder, exist_ok=True)
#
# # 复制文件夹结构
# copy_folder_structure(source_folder, destination_folder)

img1 = cv2.imread( r"E:\files\ComputerScience\Programs\image_registration\results\predict_re\20230315_p1_1_moved_1.png")
img2 = cv2.imread(r"E:\files\ComputerScience\Programs\image_registration\results\predict_re\20230315_p1_1_moved_2.png")
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img1[i,j,0] != img2[i,j,0]:
            print("！")
