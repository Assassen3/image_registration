import os
from PIL import Image
from tqdm import tqdm

# 指定输入图片所在的目录
input_dir = ".\\data\\tomato"

# 指定输出图片的目录
output_dir = ".\\data\\tomato_modified"

# 指定目标分辨率、
# TODO crop image instead of resize
target_size = (400, 208)

# 获取输入目录中的所有图片文件
image_files_ms = [f[:-6] for f in os.listdir(input_dir) if f.endswith("ms.png")]

progress_bar_ms = tqdm(total=len(image_files_ms), unit="image")

for filename in image_files_ms:
    filepath = os.path.join(input_dir, filename)

    with Image.open(filepath + 'ms.png') as ms_img:
        resized_img = ms_img.resize(target_size)
        output_filepath = os.path.join(output_dir, filename + 'ms.png')
        resized_img.save(output_filepath)

    with Image.open(filepath + 'rgb.png') as rgb_img:
        flipped_img = rgb_img.transpose(Image.ROTATE_270)
        crop_box = (142, 466, 662, 779)
        cropped_img = flipped_img.crop(crop_box)
        resized_img = cropped_img.resize(target_size)
        output_filepath = os.path.join(output_dir, filename + 'rgb.png')
        resized_img.save(output_filepath)

    with Image.open(filepath + 'depth.png') as d_img:
        flipped_img = d_img.transpose(Image.ROTATE_270)
        crop_box = (142, 466, 662, 779)
        cropped_img = flipped_img.crop(crop_box)
        resized_img = cropped_img.resize(target_size)
        output_filepath = os.path.join(output_dir, filename + 'depth.png')
        resized_img.save(output_filepath)

    # 更新进度条
    progress_bar_ms.update(1)
    progress_bar_ms.set_postfix({"Image": filename})


print("图片批量处理完成!")
