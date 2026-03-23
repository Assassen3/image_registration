import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import cpp_module


class Registrator:
    """
    Handles registration and transformation of RGBD and multispectral sensor data.

    This class manages the registration between RGBD cameras and multispectral
    sensors by loading intrinsic and extrinsic calibration parameters, computing
    transformation matrices, generating point clouds from depth and color data,
    projecting multispectral information onto point clouds, and exporting data in
    NeRF-compatible format. It maintains calibration data for both sensor types
    and provides methods to transform sensor measurements into a common coordinate
    system with configurable height, radial, and rotational adjustments.

    :ivar rgbd_intrinsic: Camera intrinsic matrix for the RGBD sensor
    :type rgbd_intrinsic: numpy.ndarray
    :ivar ms_intrinsic: Camera intrinsic matrix for the multispectral sensor
    :type ms_intrinsic: numpy.ndarray
    :ivar rgbd_extrinsic: Extrinsic transformation matrix for the RGBD sensor
    :type rgbd_extrinsic: numpy.ndarray
    :ivar ms_extrinsic: Extrinsic transformation matrix for the multispectral
        sensor
    :type ms_extrinsic: numpy.ndarray
    :ivar height: Reference height adjustment value from calibration data
    :type height: float
    :ivar radial: Reference radial adjustment value from calibration data
    :type radial: float
    """

    def __init__(self):
        self.rgbd_intrinsic = np.load('data/config/intrinsic-rgbd.npy')
        self.ms_intrinsic = np.load('data/config/intrinsic-ms.npy')
        rgbd_ex = np.load('data/config/extrinsic-rgbd.npz', allow_pickle=True)
        ms_ex = np.load('data/config/extrinsic-ms.npz', allow_pickle=True)
        omega = 2 / 180 * np.pi
        bias_rotarion = np.array([
            [np.cos(omega), np.sin(omega), 0, 0],
            [-np.sin(omega), np.cos(omega), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.rgbd_extrinsic = rgbd_ex['matrix']
        self.ms_extrinsic = bias_rotarion @ ms_ex['matrix']
        self.height = rgbd_ex['height'].item()
        self.radial = rgbd_ex['radial'].item()
        assert rgbd_ex['height'] == ms_ex['height']
        assert rgbd_ex['radial'] == ms_ex['radial']

    def get_extra_transform(self, file: Path | str):
        config = pd.read_excel(file)
        num = config.shape[0]
        height = np.array(config['HeightAdjust'] - config['PlantElevation']) / 1000.0 - self.height
        radial = -np.array(config['RadialAdjust']) / 1000.0 + self.radial
        rotation = np.array(config['DegreeRotation']) / 180 * np.pi
        extra_transform = np.stack([np.eye(4)] * num, axis=0)
        extra_transform[:, 0, 0] = np.cos(rotation)
        extra_transform[:, 0, 1] = -np.sin(rotation)
        extra_transform[:, 1, 0] = np.sin(rotation)
        extra_transform[:, 1, 1] = np.cos(rotation)
        T_trans = np.eye(4)
        T_trans[2, -1] = height[0]
        T_trans[1, -1] = radial[0]
        extra_transform = extra_transform @ T_trans
        return extra_transform

    def get_rgb_pc(self, rgb, depth, extra_mtx, offset: float = 0.0, save=False, save_path: Path = None):
        num = rgb.shape[0]

        intrinsic = self.rgbd_intrinsic
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        H, W = depth[0].shape

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()

        pcs = []
        for i in range(num):
            Z = depth[i].flatten() / 1000.0 + offset

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pos = np.vstack((X, Y, Z)).T
            pos = np.hstack([pos, np.ones_like(pos[:, 0:1])])
            pos = (extra_mtx[i] @ np.linalg.inv(self.rgbd_extrinsic) @ pos.T).T
            mask = (pos[:, 0] > -0.3) & (pos[:, 0] < 0.3) & \
                   (pos[:, 1] > -0.3) & (pos[:, 1] < 0.3) & \
                   (pos[:, 2] > -0.3) & (pos[:, 2] < 0.5)
            mask = np.ones_like(mask)

            points = pos[mask, :3]
            colors = rgb[i].reshape((-1, 4))[mask, :3] / 255.0
            pcs.append(np.hstack((points, colors)))

            if save:
                pc = np.hstack([points, (colors * 255).astype(np.uint8)])
                cpp_module.save_points(str(save_path / f'{i + 1}.ply'), pc,
                                       ['x', 'y', 'z', 'Red', 'Green', 'Blue'],
                                       ['float'] * 3 + ['uint8'] * 3)
        return pcs

    def get_ms_pc(self, rgb_pcs, ms, extra_mtx, save=False, save_path: Path = None):
        num = len(rgb_pcs)
        pcs = []
        for i in range(num):
            ms_color, mask = self.project(rgb_pcs[i], ms[i], extra_mtx[i])
            pc = np.hstack([rgb_pcs[i][:, :3][mask], (rgb_pcs[i][:, 3:][mask] * 255).astype(np.uint8), ms_color])
            pcs.append(pc)
            if save:
                save_path.mkdir(parents=True, exist_ok=True)
                cpp_module.save_points(str(save_path / f'{i + 1}.ply'), pc,
                                       ['x', 'y', 'z', 'Red', 'Green', 'Blue', 'MS'],
                                       ['float'] * 3 + ['uint8'] * 3 + ['float'])
        return pcs

    def export_nerf_json(self, rgb, depth, extra_mtx, save_path, ngp_mode=False):
        num = rgb.shape[0]
        save_path.mkdir(parents=True, exist_ok=True)
        img_save_path = save_path / 'images'
        img_save_path.mkdir(parents=True, exist_ok=True)
        for i in range(num):
            img = Image.fromarray(rgb[i])
            img.save(img_save_path / f'{i:03d}.png')
            # j = (i - 1) % num
            # mask = np.sum(rgb[i] / 255.0 - rgb[j] / 255.0, axis=-1) > 0.04
            # mask = (mask * 255.0).astype(np.uint8)
            # Image.fromarray(depth[i]).save(img_save_path / f'depth{i:03d}.png')
            # Image.fromarray(mask).save(img_save_path / f"mask{i:03d}.png")
        h, w = rgb.shape[1:3]
        fx, fy = self.rgbd_intrinsic[0, 0], self.rgbd_intrinsic[1, 1]
        trans = {
            "camera_angle_x": 2 * np.atan(w / 2 / fx),
            "camera_angle_y": 2 * np.atan(h / 2 / fy),
            "fl_x": fx,
            "fl_y": fy,
            "cx": self.rgbd_intrinsic[0, 2],
            "cy": self.rgbd_intrinsic[1, 2],
            "w": w,
            "h": h,
        }
        frames = []
        for i in range(num):
            c2w = extra_mtx[i] @ np.linalg.inv(self.rgbd_extrinsic)
            if ngp_mode:
                c2w[:3, 1:3] *= -1
            filename = f"./images/{i:03d}.png"
            frames.append({"file_path": filename, "transform_matrix": c2w.tolist(),
                           # "depth_file_path": f"./images/depth{i:03d}.png",
                           # "mask_path": f"./images/mask{i:03d}.png"
                           })
        trans['frames'] = frames
        with open(save_path / 'transforms.json', 'w', encoding='utf-8') as f:
            json.dump(trans, f)

    def project(self, pc, img, extra_mtx_single):
        pcT = np.vstack([pc[:, :3].T, np.ones_like(pc.T[0:1, :])])
        uv = self.ms_extrinsic @ np.linalg.inv(extra_mtx_single) @ pcT
        uv = self.ms_intrinsic @ (uv[:3, :] / uv[2, :])
        u, v = uv[0, :].astype(np.int64), uv[1, :].astype(np.int64)
        mask = (u > 0) & (v > 0) & (u < 2048) & (v < 1088)
        u, v = u[mask], v[mask]
        ms_color = img[v, u] / 255.0

        return ms_color[:, np.newaxis], mask


if __name__ == '__main__':
    registrator = Registrator()
    base = Path('data/60')
    extra_mtx = registrator.get_extra_transform(list(base.glob('*_v*.xlsx'))[0])
    num = extra_mtx.shape[0]
    rgb = []
    depth = []
    ms = []
    for i in range(num):
        rgb.append(np.array(Image.open(base / str(i + 1) / f'{i + 1}_color_uint8.png')))
        depth.append(np.array(Image.open(base / str(i + 1) / f'{i + 1}_depth_uint16.png')))
        ms.append(np.array(Image.open(base / str(i + 1) / f'{i + 1}.tiff')))
    rgb_np = np.array(rgb)
    depth_np = np.array(depth)
    ms = np.array(ms)
    save_path = base / 'pc'

    pcs = registrator.get_rgb_pc(rgb_np, depth_np, extra_mtx, offset=0.025, save=False)
    # registrator.get_ms_pc(pcs, ms, extra_mtx, save=True, save_path=save_path)
    registrator.export_nerf_json(rgb_np, depth_np, extra_mtx, Path('nerf'), ngp_mode=False)
