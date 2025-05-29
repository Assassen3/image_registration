import os

import numpy as np
import pandas as pd
import csv

cam2world = np.array([[0.978813707829, -0.151365548372, -0.137884676456, 0.144190746048],
                      [0.010250490159, -0.636350452900, 0.771331965923, -0.468420714817],
                      [-0.204496055841, -0.756403684616, -0.621316969395, 0.358610486366],
                      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])


def T2T(path: str):
    fs = os.listdir(path)
    for f in fs:
        if f.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(path, f))  # 读取xlsx
            matrices = []

            for _, row in df.iterrows():
                degree = - int(row['DegreeRotation'])

                theta_z = np.deg2rad(degree)

                Rz = np.array([
                    [np.cos(theta_z), -np.sin(theta_z), 0, 0],
                    [np.sin(theta_z), np.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])

                matrices.append(cam2world @ Rz)

            with open(os.path.join(path, "T.csv"), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for mat in matrices:
                    writer.writerow([mat[i // 4, i % 4] for i in range(16)])

            print('已输出 output.csv，每行是一个齐次变换矩阵的16个元素')
            break


if __name__ == '__main__':
    import numpy as np

    for i in range(5):
        degree = - int(15 * i)

        theta_z = np.deg2rad(degree)

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0, 0],
            [np.sin(theta_z), np.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        print(f'{i} ----------------------')
        print(cam2world @ Rz)


