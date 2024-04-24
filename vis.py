from PIL import Image


def blend_images(image_path1, image_path2, output_path, alpha=0.5):
    """
    将两张图片按照指定的透明度重叠在一起。

    参数:
    - image_path1: 第一张图片的路径。
    - image_path2: 第二张图片的路径。
    - output_path: 结果图片的保存路径。
    - alpha: 第一张图片的透明度，范围从0到1（0是完全透明，1是完全不透明）。
    """
    # 打开图片
    image1 = Image.open(image_path1).convert("RGBA")
    image2 = Image.open(image_path2).convert("RGBA")

    # 获取两张图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size
    point1 = [260, 650, 493, 751]
    point2 = [93, 128, 276, 198]
    resized_image = image2.resize((int(width2 / (point2[2] - point2[0]) * (point1[2] - point1[0])),
                                   int(height2 / (point2[3] - point2[1]) * (point1[3] - point1[1]))))
    print("resize: " + str(int(width2 / (point2[2] - point2[0]) * (point1[2] - point1[0]))) + ', ' + str(
        int(height2 / (point2[3] - point2[1]) * (point1[3] - point1[1]))))
    # 创建一个新的透明背景图片，尺寸等于两张图片的最大宽度和高度
    point3 = [int(point2[0] / (point2[2] - point2[0]) * (point1[2] - point1[0])),
              int(point2[1] / (point2[3] - point2[1]) * (point1[3] - point1[1]))]

    result_image = Image.new('RGBA', (width1, height1), (0, 0, 0, 0))
    result_image.paste(resized_image, (point1[0] - point3[0], point1[1] - point3[1]))
    print("box: " + str(point1[0] - point3[0]) + ',' + str(point1[1] - point3[1]))
    # 按照指定的透明度混合图片
    blended_image = Image.blend(image1, result_image, alpha=alpha)

    # 保存混合后的图片
    blended_image.save(output_path)


# 使用示例
blend_images('./results/rgb.png', './results/ms.png', './results/results.png', alpha=0.6)
