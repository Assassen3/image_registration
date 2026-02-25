import cv2
import numpy as np
from PIL import Image, ImageDraw

import cpp_module


def check_axis_point(u, v, depth_np, offset, extra_mtx, registrater, save_path):
    """
    Validates and visualizes axis points by transforming depth data through
    camera intrinsics and extrinsics, then saves the result as a point cloud.

    This function processes depth measurements at a specific pixel coordinate
    (u, v) across multiple frames, converting them from image space to 3D world
    space using camera calibration parameters. The transformed points are
    colorized based on their frame index and saved as a PLY file for
    visualization.

    :param u: Horizontal pixel coordinate in the depth image
    :type u: int
    :param v: Vertical pixel coordinate in the depth image
    :type v: int
    :param depth_np: Array of depth maps with shape (num_frames, height, width)
        containing depth values in millimeters
    :type depth_np: numpy.ndarray
    :param offset: Depth offset value in meters to be added to converted depth
        measurements
    :type offset: float
    :param extra_mtx: Transformation matrices for each frame with shape
        (num_frames, 4, 4) to apply additional transformations
    :type extra_mtx: numpy.ndarray
    :param registrater: Object containing camera calibration parameters including
        rgbd_intrinsic and rgbd_extrinsic matrices
    :param save_path: Directory path where the output PLY file will be saved
    :type save_path: pathlib.Path
    :return: None
    :rtype: None

    :notes:
        The function converts depth from millimeters to meters, applies camera
        intrinsics to transform from image coordinates to camera space, then
        applies extrinsics and extra transformations to move to world space.
        Colors are assigned using a gradient from red to cyan based on frame
        index.
    """
    num = extra_mtx.shape[0]
    l = []
    c = []
    for i in range(num):
        Z = depth_np[i, v, u] / 1000.0 + offset
        fx = registrater.rgbd_intrinsic[0, 0]
        fy = registrater.rgbd_intrinsic[1, 1]
        cx = registrater.rgbd_intrinsic[0, 2]
        cy = registrater.rgbd_intrinsic[1, 2]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pos = np.vstack((X, Y, Z)).T
        pos = np.hstack([pos, np.ones_like(pos[:, 0:1])])
        pos = (extra_mtx[i] @ registrater.rgbd_extrinsic @ pos.T).T

        points = pos[:, :3]
        colors = np.array([1.0, 1.0 / num * i, 1.0])
        l.append(points)
        c.append(colors)
    points = np.vstack(l)
    colors = np.vstack(c)
    pc = np.hstack([points, (colors * 255).astype(np.uint8)])
    cpp_module.save_points(str(save_path / f'check2.ply'), pc,
                           ['x', 'y', 'z', 'Red', 'Green', 'Blue'],
                           ['float'] * 3 + ['uint8'] * 3)


def visualize_results(registrater, rgb_np, extra_mtx, save_path):
    """
    Visualizes registration results by projecting 3D coordinate axes onto RGB images and saving them.

    This function takes a batch of RGB images and their corresponding transformation matrices,
    projects 3D coordinate system axes (X, Y, Z) onto each image using camera parameters from
    the registrater, and saves the annotated images with colored axis lines overlaid. The axes
    are drawn in red (X), green (Y), and blue (Z) to visualize the spatial orientation and
    registration quality.

    :param registrater: Object containing camera parameters including rgbd_extrinsic and
                         rgbd_intrinsic matrices for coordinate transformations
    :param rgb_np: Batch of RGB images as a numpy array with shape (N, H, W, C) where N is
                   the number of images
    :param extra_mtx: Array of additional transformation matrices with shape (N, 4, 4) for
                      each corresponding image
    :param save_path: Directory path where the annotated images will be saved
    :type save_path: pathlib.Path
    :return: None
    """
    num = rgb_np.shape[0]
    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(num):
        axis_points = np.array([[0, 0, 0, 1],
                                [0.05, 0, 0, 1],
                                [0, 0.05, 0, 1],
                                [0, 0, 0.05, 1]], dtype=np.float32)
        axis_points_camera = np.linalg.inv(registrater.rgbd_extrinsic) @ np.linalg.inv(extra_mtx[i]) @ axis_points.T
        axis_points_camera = registrater.rgbd_intrinsic @ (axis_points_camera[:3, :] / axis_points_camera[2, :])
        image = Image.fromarray(rgb_np[i])
        draw = ImageDraw.Draw(image)
        draw.line(([axis_points_camera[0, 0], axis_points_camera[1, 0]],
                   [axis_points_camera[0, 1], axis_points_camera[1, 1]]), fill="red", width=2)
        draw.line(([axis_points_camera[0, 0], axis_points_camera[1, 0]],
                   [axis_points_camera[0, 2], axis_points_camera[1, 2]]), fill="green", width=2)
        draw.line(([axis_points_camera[0, 0], axis_points_camera[1, 0]],
                   [axis_points_camera[0, 3], axis_points_camera[1, 3]]), fill="blue", width=2)
        print(
            f"{axis_points_camera[0, 0]:5f}   {axis_points_camera[1, 0]:5f}    |   {axis_points_camera[0, 3]:5f}    {axis_points_camera[1, 3]:5f}")
        image.save(save_path / f'{i}.png')


def draw(img, corners, imgpts):
    """
    Draws three perpendicular axes originating from a corner point on an image.

    Visualizes a 3D coordinate system by drawing three colored lines (red, green,
    and blue) from a reference corner point to three projected axis endpoints. The
    lines represent the X, Y, and Z axes in the 3D space projected onto the 2D
    image plane.

    :param img: Input image on which the axes will be drawn
    :param corners: Array of corner points where the first corner is used as the
                    origin point for the axes
    :param imgpts: Array of three 3D points projected onto the image plane
                   representing the endpoints of the X, Y, and Z axes
    :return: Modified image with the three axis lines drawn from the corner point
    ```"""
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 2)
    # axis = np.float32([[0.15, 0, 0], [0, 0.15, 0], [0, 0, -0.15]]).reshape(-1, 3)
    # i = 0
    # for rvec, tvec, img in zip(rvecs, tvecs, all_imgs):
    #     zeros, _ = cv2.projectPoints(np.zeros_like(axis), rvec, tvec, mtx, dist)
    #     imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    #     img = draw(img, zeros, imgpts)
    #     cv2.imwrite(str(save_path / f'{i:03d}.png'), img)
    #     i += 1
    return img
