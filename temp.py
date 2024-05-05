import open3d as o3d

color_raw = o3d.t.io.read_image(r"results/20230315_p1_3_rgb.png")
depth_raw = o3d.t.io.read_image(r"results/20230315_p1_3_depth.png")
depth_raw = depth_raw.to(dtype=o3d.core.Dtype.UInt16)
# rgbd_image = o3d.t.geometry.RGBDImage(color_raw, depth_raw)
# print(rgbd_image)


# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()
# intrinsic = o3d.camera.PinholeCameraIntrinsic(1024, 1024,
#                                               0.49117687344551086,
#                                               0.49127498269081116,
#                                               0.50117456912994385,
#                                               0.50129920244216919
#                                               )
intrinsic = o3d.core.Tensor([[0.49117687344551086, 0, 0.50117456912994385],
                             [0, 0.49127498269081116, 0.50129920244216919],
                             [0, 0, 1]])
# intrinsic = o3d.core.Tensor([[491.17687344551086, 0, 501.17456912994385],
#                              [0, 491.27498269081116, 501.29920244216919],
#                              [0, 0, 1]])
# intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
#                              [0, 0, 1]])
pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth=depth_raw, intrinsics=intrinsic,depth_scale=500000.0,)
o3d.visualization.draw([pcd])

print(1)  # 输出点云点的个数
