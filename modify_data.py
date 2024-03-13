import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 指定输入图片所在的目录
input_dir = ".\\data\\tomato"

# 指定输出图片的目录
output_dir = ".\\data\\tomato_modified"

# 指定目标分辨率
target_size = (384, 192)

# 获取输入目录中的所有图片文件
image_files_ms = [f for f in os.listdir(input_dir) if f.endswith("ms.png")]

progress_bar_ms = tqdm(total=len(image_files_ms), unit="image")

for filename in image_files_ms:
    filepath = os.path.join(input_dir, filename)

    try:
        # 打开图片
        with Image.open(filepath) as img:

            resized_img = img.resize(target_size)
            output_filepath = os.path.join(output_dir, filename)

            resized_img.save(output_filepath)


    except Exception as e:
        print(f"处理图片 {filename} 时出错: {str(e)}")

    # 更新进度条
    progress_bar_ms.update(1)
    progress_bar_ms.set_postfix({"Image": filename})

# 关闭进度条
progress_bar_ms.close()

image_files_rgb = [f for f in os.listdir(input_dir) if f.endswith("rgb.png")]
progress_bar_rgb = tqdm(total=len(image_files_rgb), unit="image")
for filename in image_files_rgb:
    filepath = os.path.join(input_dir, filename)

    try:
        # 打开图片
        with Image.open(filepath) as img:
            flipped_img = img.transpose(Image.ROTATE_270)
            crop_box = (142, 466, 662, 779)
            cropped_img = flipped_img.crop(crop_box)
            resized_img = cropped_img.resize(target_size)
            output_filepath = os.path.join(output_dir, filename)

            resized_img.save(output_filepath)


    except Exception as e:
        print(f"处理图片 {filename} 时出错: {str(e)}")

    # 更新进度条
    progress_bar_rgb.update(1)
    progress_bar_rgb.set_postfix({"Image": filename})

print("图片批量处理完成!")
