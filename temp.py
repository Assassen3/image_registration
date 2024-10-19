import open3d as o3d

# 设置 ICP 配准参数
threshold = 0.05
trans_init = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]  # 使用 IMU 提供的初始变换矩阵
print("初始变换矩阵：", trans_init)
# 使用 ICP 进行点云配准

point_cloud_1 = o3d.io.read_point_cloud("3.ply")
point_cloud_2 = o3d.io.read_point_cloud("4.ply")

# o3d.io.write_point_cloud(R"D:\files\ComputerScience\Programs\image_registration\4.ply", point_cloud_2, write_ascii= True)

if point_cloud_1.is_empty() or point_cloud_2.is_empty():
    print("One of the point clouds is empty!")

print("Initial transformation matrix:\n", trans_init)

reg_p2p = o3d.pipelines.registration.registration_icp(
    point_cloud_1, point_cloud_2, threshold, trans_init
)

# 输出变换矩阵
print("ICP 变换矩阵:")
print(reg_p2p.transformation)

# 将点云1转换到点云2的坐标系
# point_cloud_1.transform(reg_p2p.transformation)
# print("完成坐标系转换")
# 合并两个彩色点云
# combined_point_cloud = point_cloud_1 + point_cloud_2

# 显示合并的彩色点云
# o3d.visualization.draw_geometries([combined_point_cloud],
#                                   window_name="Aligned Colored Point Clouds",
#                                   point_show_normal=False)
