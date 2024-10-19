import open3d as o3d
import numpy as np

point_cloud_1 = o3d.io.read_point_cloud("3.ply")
point_cloud_2 = o3d.io.read_point_cloud("4.ply")

points = np.asarray(point_cloud_1.points)

# 筛选出不包含 (0, 0, 0) 点的点云
filtered_points = points[~np.all(points == [0, 0, 0], axis=1)]

# 创建一个新的点云对象
filtered_pcd = o3d.geometry.PointCloud()

# 将过滤后的点云数组赋值给新的点云对象
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# 保存或可视化去除 (0, 0, 0) 点后的点云
o3d.io.write_point_cloud("5.ply", filtered_pcd)
o3d.visualization.draw_geometries([filtered_pcd])
