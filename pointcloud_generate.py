import csv
import os
import cv2
import k4a
import numpy as np


def processCSVFile(csv_file):
    matrices_ = []  # 存储矩阵的列表
    # 逐行读取CSV文件
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row_data = [float(value) for value in row]

            # 检查每行是否包含16个元素，然后将其转换为4x4矩阵
            if len(row_data) != 16:
                print("Error: CSV row in", csv_file, "does not contain 16 elements.")
                continue
            # 创建4x4矩阵并填充数据
            matrix = np.array(row_data).reshape((4, 4))
            matrices_.append(matrix)
    return matrices_


baseFolder = "data/tomato"
csv_files = ["3d_reconstruction_for_greenhouse_tomato/UR_poses_data.csv",
             "3d_reconstruction_for_greenhouse_tomato/T_230315_v3.csv",
             "3d_reconstruction_for_greenhouse_tomato/T_230705_v3.csv"]

T_a2b = np.array([[0, -1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

all_matrices = []  # 存储所有矩阵的列表

for csv_file in csv_files:
    print("Processing", csv_file + ":")
    matrices = processCSVFile(csv_file)

    print("Number of matrices in", csv_file + ":", len(matrices))
    all_matrices.append(matrices)

data_path = "data/tomato"
filenames = [f[:-6] for f in os.listdir(data_path) if f.endswith("ms.png")]
for filename in filenames:
    year_number = int(filename[5:8])
    p_number = int(filename[10])
    print('Processing: year: ' + str(year_number) + '  p: ' + str(p_number))

    # 根据数字选择第一个矩阵
    selected_matrix1 = None
    selected_matrix1_col = -1
    selected_matrix1_row = 1 if year_number == 315 or year_number == 403 else 2

    # 根据数字选择第二个矩阵
    selected_matrix2 = None
    selected_matrix2_row = 0  # 默认选择第一页
    selected_matrix2_col = p_number - 1
    selected_matrix2 = all_matrices[0][p_number - 1]

    for R in range(6):
        for N in range(1, 25):
            i = 24 * R + N

            if p_number == 1 or p_number == 4 or p_number == 5 or p_number == 7 or p_number == 9:
                if N == 1:
                    # 如果 N 等于 1，根据年份选择矩阵
                    selected_matrix1 = all_matrices[1][23 * R + N - 1] if year_number == 315 or year_number == 403 else \
                        all_matrices[2][23 * R + N - 1]
                    selected_matrix1_col = 23 * R + N - 1
                else:
                    # 如果 N 不等于 1，根据 N 的不同值选择矩阵
                    selected_matrix1 = all_matrices[1][23 * R + N - 2] if year_number == 315 or year_number == 403 else \
                        all_matrices[2][23 * R + N - 2]
                    selected_matrix1_col = 23 * R + N - 2
                print("选择的 T_greenhouse2base 在 all_matrices 中的索引：第", selected_matrix1_row + 1, "页，第",
                      selected_matrix1_col + 1, "个矩阵")
            else:
                if N == 24:
                    # 如果 N 等于 1，根据年份选择矩阵
                    selected_matrix1 = all_matrices[1][23 * R + N - 2] if year_number == 315 or year_number == 403 else \
                        all_matrices[2][23 * R + N - 2]
                    selected_matrix1_col = 23 * R + N - 2
                else:
                    # 如果 N 不等于 1，根据 N 的不同值选择矩阵
                    selected_matrix1 = all_matrices[1][23 * R + N - 1] if year_number == 315 or year_number == 403 else \
                        all_matrices[2][23 * R + N - 1]
                    selected_matrix1_col = 23 * R + N - 1
                print("选择的 T_greenhouse2base 在 all_matrices 中的索引：第", selected_matrix1_row + 1, "页，第",
                      selected_matrix1_col + 1, "个矩阵")

            # Load calibration data (you may want to modify this logic)
            with open(os.path.join(data_path, filename + 'calibration.json'), 'rb') as fin:
                buffer = bytearray(fin.read())
                buffer.append(0)

            calibration = k4a.Calibration.create_from_raw(
                buffer,
                k4a.EDepthMode.NFOV_UNBINNED,
                k4a.EColorResolution.RES_720P)

            transformation = k4a.Transformation(calibration)

            rgb_img = cv2.imread(os.path.join(data_path, filename + 'rgb.png'), cv2.IMREAD_UNCHANGED)
            depth_img = cv2.imread(os.path.join(data_path, filename + 'depth.png'), cv2.IMREAD_UNCHANGED)
            rgb_Img = k4a.Image.create(k4a.EImageFormat.COLOR_BGRA32, rgb_img.shape[1], rgb_img.shape[0],
                                       rgb_img.shape[1] * 4)
            rgb_Img._data = rgb_img
            depth_Img = k4a.Image.create(k4a.EImageFormat.DEPTH16, depth_img.shape[1], depth_img.shape[0],
                                         depth_img.shape[1] * 2)
            depth_Img._data = depth_img
            pc_Image = transformation.depth_image_to_point_cloud(depth_Img, k4a.ECalibrationType.COLOR)
            pc_image = pc_Image._data.copy()
            pc_image = pc_image.reshape((-1, 3))
            c = 0
            for point in pc_image:
                if sum(point) != 0:
                    c += 1
            print(c)
