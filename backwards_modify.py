import os

from PIL import Image
from tqdm import tqdm

# 指定输入图片所在的目录
input_dir = ".\\results\\predict"

# 指定输出图片的目录
output_dir = ".\\data\\tomato_dn_back_modified"

# 指定目标分辨率
# TODO crop image instead of resize
target_size = (400, 208)

# 获取输入目录中的所有图片文件
image_files_ms = [f[:-12] for f in os.listdir(input_dir) if f.endswith("_moved_1.png")]

progress_bar_ms = tqdm(total=len(image_files_ms), unit="image")

for filename in image_files_ms:
    for i in range(25):
        filepath = os.path.join(input_dir, filename + "_moved_" + str(i + 1) + ".png")
        crop_box = (142, 466, 662, 779)

        with Image.open(filepath) as ms_img:
            restored_cropped_img = ms_img.resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]))
            restored_img = Image.new('RGB', (720, 1280), (255, 255, 255))  # 使用白色填充空白区域
            restored_img.paste(restored_cropped_img, crop_box)
            restored_img = restored_img.transpose(Image.ROTATE_90)
            output_filepath = os.path.join(output_dir, filename + "_ms_" + str(i + 1) + ".png")
            restored_img.save(output_filepath)
    # 更新进度条
    progress_bar_ms.update(1)
    progress_bar_ms.set_postfix({"Image": filename})

print("图片批量处理完成!")
