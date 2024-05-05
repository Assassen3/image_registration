import matplotlib.pyplot as plt


def plot_history(hist):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(hist.epoch, hist.history['loss'], '.-')
    axs[0, 0].set_title('Total Loss')

    axs[0, 1].plot(hist.epoch, hist.history['output_1_loss'], '.-')
    axs[0, 1].set_title('SSIM')

    axs[1, 0].plot(hist.epoch, hist.history['output_2_loss'], '.-')
    axs[1, 0].set_title('Smooth Loss')

    axs[1, 1].plot(hist.epoch, hist.history['output_3_loss'], '.-')
    axs[1, 1].set_title('SSIM with Depth')

    fig.suptitle('Loss')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(hist.epoch, hist.history['output_1_normalized_mutual_information'], '.-')
    axs[0, 0].set_title('NMI')

    axs[0, 1].plot(hist.epoch, hist.history['output_1_correlation_coefficient'], '.-')
    axs[0, 1].set_title('cc')

    axs[1, 0].plot(hist.epoch, hist.history['output_1_calculate_ssim'], '.-')
    axs[1, 0].set_title('SSIM')

    fig.suptitle('Metrics')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import open3d as o3d
    import os

    # 设置点云文件夹路径
    folder_path = r'E:\files\毕业设计\tomato_data_organs_pose\20230315\p1'

    # 获取文件夹中所有点云文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('pc.pcd')]

    # 读取并存储所有点云
    point_clouds = []
    for file in files:
        # 读取点云文件
        cloud = o3d.io.read_point_cloud(file)
        point_clouds.append(cloud)
        print(f"Loaded {file}")

    # 可视化所有读取的点云
    o3d.visualization.draw_geometries(point_clouds)
