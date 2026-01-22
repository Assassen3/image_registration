from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class Calibrator:
    def __init__(self):
        self.board = cv2.aruco.GridBoard(
            (8, 5),
            0.04,
            0.01,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )
        self.detector = cv2.aruco.ArucoDetector()

    def get_corners(self, img_list: list[np.ndarray]):
        all_corners = []
        all_ids = []
        for img in img_list:
            corners, ids, rejected_points = self.detector.detectMarkers(img)
            corners, ids, _, _ = self.detector.refineDetectedMarkers(img, self.board, corners, ids, rejected_points)
            corners = np.concat(corners, axis=0)
            indices = ids.argsort(axis=0)[:, 0]
            corners = corners[indices]
            ids = ids[indices]
            all_corners.append(corners)
            all_ids.append(ids)

        dst_path = Path(__file__).parent / 'output'
        dst_path.mkdir(parents=True, exist_ok=True)
        for idx, corners in enumerate(all_corners):
            draw_img = img_list[idx].copy()
            cv2.drawChessboardCorners(draw_img, (8, 5), corners[:, 0, :].copy(), True)
            cv2.imwrite(str(dst_path / f'{idx:03d}.png'), draw_img)

        return all_corners, all_ids

    def calibrate(self, img_list: list[np.ndarray]):
        img_shape = img_list[0].shape[:2][::-1]
        all_corners, all_ids = self.get_corners(img_list)
        all_obj_points = []
        all_img_points = []
        for corners, ids in zip(all_corners, all_ids):
            obj_points, img_points = self.board.matchImagePoints(corners, ids, None, None)
            all_obj_points.append(obj_points.squeeze())
            all_img_points.append(img_points.squeeze())

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, img_shape,
                                                           None, None,
                                                           flags=(cv2.CALIB_FIX_K1 |
                                                                  cv2.CALIB_FIX_K2 |
                                                                  cv2.CALIB_FIX_K3 |
                                                                  cv2.CALIB_ZERO_TANGENT_DIST))

        return ret, mtx, dist, rvecs, tvecs

    def compute_turntable_transform(self, tvecs):
        points = np.array([t.flatten() for t in tvecs])
        n_points = points.shape[0]

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        U, S, Vt = np.linalg.svd(centered_points)
        normal = Vt[2, :]
        normal = normal if normal[-1] < 0 else -normal
        z_new = normal / np.linalg.norm(normal)

        p = np.cross(z_new, np.array([1, 0, 0]))
        p = p / np.linalg.norm(p)
        q = np.cross(z_new, p)

        u_coords = np.dot(centered_points, p)
        v_coords = np.dot(centered_points, q)

        # Equation: u^2 + v^2 + D*u + E*v + F = 0
        # Center (uc, vc) = (-D/2, -E/2)
        A = np.column_stack((u_coords, v_coords, np.ones(n_points)))
        b = -(u_coords ** 2 + v_coords ** 2)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = x
        uc = -D / 2
        vc = -E / 2

        origin_new = centroid + uc * p + vc * q

        camera_center = np.array([0.0, 0.0, 0.0])
        vec_center_to_cam = camera_center - origin_new

        # v_proj = v - (v . z) * z
        dist_along_normal = np.dot(vec_center_to_cam, z_new)
        vec_proj = vec_center_to_cam - dist_along_normal * z_new
        norm_proj = np.linalg.norm(vec_proj)
        y_new = vec_proj / norm_proj
        x_new = np.cross(y_new, z_new)
        x_new = x_new / np.linalg.norm(x_new)

        R_new_to_cam = np.column_stack((x_new, y_new, z_new))
        t_new_to_cam = origin_new.reshape(3, 1)
        R_cam_to_new = R_new_to_cam.T
        t_cam_to_new = -np.dot(R_cam_to_new, t_new_to_cam)

        T_cam_to_new = np.eye(4)
        T_cam_to_new[:3, :3] = R_cam_to_new
        T_cam_to_new[:3, 3] = t_cam_to_new.flatten()

        return T_cam_to_new


def draw(img, corners, imgpts):
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


if __name__ == '__main__':
    cali = Calibrator()
    cali_base = Path(__file__).parent / 'data' / 'cali'
    img_paths = list(cali_base.rglob('*.tiff'))
    save_path = Path(__file__).parent / 'output'
    save_path.mkdir(parents=True, exist_ok=True)

    all_imgs = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_paths]

    ret, mtx, dist, rvecs, tvecs = cali.calibrate(all_imgs)
