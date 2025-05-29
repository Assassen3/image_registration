import Metashape
import os
import argparse
import sys


folder_path = sys.argv[1]
folder_name = sys.argv[2]
output_folder = sys.argv[3]


# 创建新的Metashape项目
doc = Metashape.app.document

# 遍历输入文件夹中的每个子文件夹


# 设置项目名称为子文件夹名称
project_name = folder_name
project_path = os.path.join(output_folder, f"{project_name}.psx")

# 保存项目
doc.save(project_path)

# 添加新的chunk
chunk = doc.addChunk()

# 加载子文件夹中的图像到chunk中
image_list = [os.path.join(folder_path, img) for img in os.listdir(folder_path)
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]

if not image_list:
    print(f"No images found in {folder_name}. Skipping...")

chunk.addPhotos(image_list)
doc.save()

# 对齐照片
chunk.matchPhotos(downscale=4, generic_preselection=True, reference_preselection=False)
chunk.alignCameras()
doc.save()

# 检查相机状态
for camera in chunk.cameras:
    if not camera.transform:
        print(f"Warning: Camera {camera.label} in {folder_name} is not aligned.")

# 生成DepthMaps
chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.AggressiveFiltering)

# 生成稠密点云
chunk.buildPointCloud(source_data=Metashape.DepthMapsData, point_colors=True, point_confidence=False,
                    keep_depth=True, max_neighbors=100, uniform_sampling=True, points_spacing=0.1)
doc.save()

# 导出点云
dense_cloud_path = os.path.join(output_folder, f"{project_name}_dense_cloud.ply")
chunk.exportPointCloud(path=dense_cloud_path,
                    source_data=Metashape.PointCloudData,
                    binary=True, save_point_color=True,
                    save_point_normal=True, format=Metashape.PointCloudFormatPLY,
                    compression=True)
print(f"Dense cloud saved as: {dense_cloud_path}")

# 保存项目文件
doc.save()
# 关闭 Metashape 项目和 chunk（优化部分）
doc.clear()  # 关闭项目（如果不再需要该项目）