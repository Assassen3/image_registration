import cv2
import numpy as np
import load_data
import tqdm


def feature_based_registration(moving_image, fixed_image):
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(moving_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(fixed_image, None)

    # 创建FLANN匹配器
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比率测试筛选良好匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配的关键点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        # 计算透视变换矩阵
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        registered_image = cv2.warpPerspective(moving_image, M, (fixed_image.shape[1], fixed_image.shape[0]))

        return registered_image

    except cv2.error as e:
        print(e)
        return None
    # 应用透视变换进行图像配准


data = load_data.load_data(100)

bar = tqdm.tqdm(total=data.shape[0])
# 读取图像
for i in range(data.shape[0]):
    moving_image = data[i, 0, ...]
    fixed_image = data[i, 1, ...]

    moving_image = (moving_image * 255).astype(np.uint8)
    fixed_image = (fixed_image * 255).astype(np.uint8)

    registered_image = feature_based_registration(moving_image, fixed_image)
    if registered_image is not None:
        cv2.imwrite('./results/sift/' + str(i) + '.png', registered_image)
    bar.update(1)
bar.close()
