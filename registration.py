import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import cpp_module


class Registrator:
    def __init__(self):
        self.rgbd_intrinsic = np.load('data/config/intrinsic-rgbd.npy')
        self.ms_intrinsic = np.load('data/config/intrinsic-ms.npy')
        rgbd_ex = np.load('data/config/extrinsic-rgbd.npz', allow_pickle=True)
        ms_ex = np.load('data/config/extrinsic-ms.npz', allow_pickle=True)
        self.rgbd_extrinsic = rgbd_ex['matrix']
        self.ms_extrinsic = ms_ex['matrix']
        self.height = rgbd_ex['height'].item()
        self.radial = rgbd_ex['radial'].item()
        assert rgbd_ex['height'] == ms_ex['height']
        assert ms_ex['radial'] == ms_ex['radial']

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

    def get_rgb_pc(self, rgb, depth, extra_mtx, save_path: Path):
        num = rgb.shape[0]
        for i in range(num):
            pos = cpp_module.registration(rgb[i], depth[i]) / 1000.0
            pos = np.hstack([pos, np.ones_like(pos[:, 0:1])])
            pos = (extra_mtx[i] @ self.rgbd_extrinsic @ pos.T).T
            mask = (pos[:, 0] > -0.3) & (pos[:, 0] < 0.3) & \
                   (pos[:, 1] > -0.3) & (pos[:, 1] < 0.3) & \
                   (pos[:, 2] > -0.5) & (pos[:, 2] < 0.5)
            pc = np.hstack([pos[mask, :3], rgb[i].reshape((-1, 4))[mask, :3]])
            cpp_module.save_points(str(save_path / f'{i + 1}.ply'), pc, ['x', 'y', 'z', 'Red', 'Green', 'Blue'],
                                   ['float'] * 3 + ['uint8'] * 3)
            print(i)

    def visualize_results(self, rgb, depth, extra_mtx, save_path):
        num = rgb.shape[0]
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
            image.save(save_path / f'{i}.png')

    def export_nerf_json(self, rgb, depth, extra_mtx, save_path):
        num = rgb.shape[0]
        save_path.mkdir(parents=True, exist_ok=True)
        img_save_path = save_path / 'images'
        img_save_path.mkdir(parents=True, exist_ok=True)
        for i in range(num):
            img = Image.fromarray(rgb[i])
            img.save(img_save_path / f'{i:03d}.png')
            j = (i - 1) % num
            mask = np.sum(rgb[i] / 255.0 - rgb[j] / 255.0, axis=-1) > 0.04
            mask = (mask * 255.0).astype(np.uint8)
            Image.fromarray(depth[i]).save(img_save_path / f'depth{i:03d}.png')
            Image.fromarray(mask).save(img_save_path / f"mask{i:03d}.png")
        h, w = rgb.shape[1:3 ]
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
            c2w = extra_mtx[i] @ self.rgbd_extrinsic @ np.diag([1, -1, -1, 1])
            filename = f"./images/{i:03d}.png"
            frames.append({"file_path": filename, "transform_matrix": c2w.tolist(),
                           "depth_file_path":f"./images/depth{i:03d}.png",
                          "mask_path": f"./images/mask{i:03d}.png"})
        trans['frames'] = frames
        with open(save_path / 'transforms.json', 'w', encoding='utf-8') as f:
            json.dump(trans, f)


if __name__ == '__main__':
    registrater = Registrator()
    extra_mtx = registrater.get_extra_transform('data/K1/T_K1_v36.xlsx')
    base = Path('data/K1')
    rgb = []
    depth = []
    for i in range(36):
        rgb.append(np.array(Image.open(base / str(i + 1) / f'{i + 1}_color_uint8.png')))
        depth.append(np.array(Image.open(base / str(i + 1) / f'{i + 1}_depth_uint16.png')))
    rgb_np = np.array(rgb)
    depth_np = np.array(depth)

    save_path = Path('data/K1/pc')
    save_path.mkdir(parents=True, exist_ok=True)
    registrater.get_rgb_pc(rgb_np, depth_np, extra_mtx, save_path)

    # registrater.export_nerf_json(rgb_np, depth_np,extra_mtx, save_path=Path('data/K1/nerf'))
