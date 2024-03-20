import os
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

def main():
    base_folder = "E:\\Tomato_data"
    csv_files = [
        "E:\\Tomato_data\\map_and_pose_data\\UR_poses_data.csv",
        "E:\\Tomato_data\\map_and_pose_data\\tomato_data_proc\\T_230315_v3.csv",
        "E:\\Tomato_data\\map_and_pose_data\\tomato_data_proc\\T_230705_v3.csv"
    ]
    T_a2b = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    all_matrices = []

    for csv_file in csv_files:
        print(f"Processing {csv_file}:")
        matrices = process_csv_file(csv_file)
        print(f"Number of matrices in {csv_file}: {len(matrices)}")
        all_matrices.append(matrices)

    is_transformed = True
    is_mkv_frames = False

    # Iterate over folders in the base folder
    for entry in os.scandir(base_folder):
        if entry.is_dir() and entry.name.startswith("2023"):
            temp_folder = entry.path
            print(f"Processing folder: {temp_folder}")

            year_number = 0
            year_start = temp_folder.find("20230")
            if year_start != -1 and year_start + 5 < len(temp_folder):
                year_number_str = temp_folder[year_start + 5:]
                year_number = int(year_number_str)
                print(f"年份数字: {year_number}")

            # Iterate over subfolders in the current folder
            for subentry in os.scandir(temp_folder):
                if subentry.is_dir() and subentry.name.startswith("p"):
                    subfolder_path = subentry.path
                    print(f"Processing subfolder: {subfolder_path}")

                    p_number = 0
                    p_start = subfolder_path.find("p")
                    if p_start != -1 and p_start + 1 < len(subfolder_path):
                        p_number_str = subfolder_path[p_start + 1:]
                        p_number = int(p_number_str)
                        print(f"p数字: {p_number}")

                    # Select first matrix based on number
                    selected_matrix1_row = 1 if year_number in [315, 403] else 2
                    selected_matrix1_col = -1

                    # Select second matrix based on number
                    selected_matrix2_row = 0
                    selected_matrix2_col = p_number - 1

                    if 1 <= p_number <= len(all_matrices[0]):
                        selected_matrix2 = all_matrices[0][p_number - 1]
                        print(f"选择的 T_base2cam 在 all_matrices 中的索引:第 {selected_matrix2_row + 1} 页,第 {selected_matrix2_col + 1} 个矩阵")

                    # Process frames in this subfolder
                    for R in range(6):
                        for N in range(1, 25):
                            i = 24 * R + N

                            if p_number in [1, 4, 5, 7, 9]:
                                if N == 1:
                                    selected_matrix1 = all_matrices[1][23 * R + N - 1] if year_number in [315, 403] else all_matrices[2][23 * R + N - 1]
                                    selected_matrix1_col = 23 * R + N - 1
                                else:
                                    selected_matrix1 = all_matrices[1][23 * R + N - 2] if year_number in [315, 403] else all_matrices[2][23 * R + N - 2]
                                    selected_matrix1_col = 23 * R + N - 2
                                print(f"选择的 T_greenhouse2base 在 all_matrices 中的索引:第 {selected_matrix1_row + 1} 页,第 {selected_matrix1_col + 1} 个矩阵")
                            else:
                                if N == 24:
                                    selected_matrix1 = all_matrices[1][23 * R + N - 2] if year_number in [315, 403] else all_matrices[2][23 * R + N - 2]
                                    selected_matrix1_col = 23 * R + N - 2
                                else:
                                    selected_matrix1 = all_matrices[1][23 * R + N - 1] if year_number in [315, 403] else all_matrices[2][23 * R + N - 1]
                                    selected_matrix1_col = 23 * R + N - 1
                                print(f"选择的 T_greenhouse2base 在 all_matrices 中的索引:第 {selected_matrix1_row + 1} 页,第 {selected_matrix1_col + 1} 个矩阵")

                            # Construct paths for color, depth, and point cloud files
                            color_file_path = os.path.join(subfolder_path, str(i), f"{i}_color_uint8_segmented.png")
                            depth_file_path = os.path.join(subfolder_path, str(i), f"{i}_depth_uint16.png")
                            calibration_file_path = os.path.join(subfolder_path, str(i), f"{i}_calibration.json")
                            save_file_path = os.path.join(subfolder_path, f"{i}_pc.pcd")
                            save_pose_txt_path = os.path.join(subfolder_path, str(i), f"{i}_pose.txt")
                            label_file_path = os.path.join(subfolder_path, str(i), "raw_label.png")
                            save_stem_pc_path = os.path.join(subfolder_path, "stems", f"{i}_stem_pc.pcd")
                            save_fruit_pc_path = os.path.join(subfolder_path, "fruits", f"{i}_fruit_pc.pcd")
                            save_leaf_pc_path = os.path.join(subfolder_path, "leaves", f"{i}_leaf_pc.pcd")
                            save_flower_pc_path = os.path.join(subfolder_path, "flowers", f"{i}_flower_pc.pcd")

                            # Load calibration data
                            with open(calibration_file_path, 'rb') as f:
                                calibration_data = f.read()

                            # Process the frame
                            frames_to_pc_v3(color_file_path, depth_file_path, label_file_path,
                                            save_stem_pc_path, save_fruit_pc_path, save_leaf_pc_path, save_flower_pc_path,
                                            is_transformed, is_mkv_frames,
                                            calibration_data, selected_matrix1, T_a2b, selected_matrix2)
                            print(f"-------- {i} -------- Saving file done!")

    print("-------- All -------- Saving files done!")
