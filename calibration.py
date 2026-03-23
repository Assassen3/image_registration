from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class Calibrator:
    def __init__(self):
        """
        Initializes the ArUco marker detection system with a predefined grid board
        configuration and detector.

        Creates an 8x5 grid board of ArUco markers using the DICT_4X4_50 dictionary
        with specified marker and separator dimensions. The grid board is used for
        camera calibration and pose estimation. Additionally, initializes an ArUco
        detector for identifying and locating markers in images.

        :raises: None
        :rtype: None
        """
        self.board = cv2.aruco.GridBoard(
            (8, 5),
            0.04,
            0.01,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )
        self.detector = cv2.aruco.ArucoDetector()

    def get_corners(self, img_list: list[np.ndarray]):
        """
        Detects and refines ArUco marker corners from a list of images,
        then saves visualizations of the detected corners.

        This method processes multiple images to detect ArUco markers, refines
        their corner positions, sorts them by marker ID, and generates debug
        images showing the detected chessboard corners. The detected corners
        are organized and returned along with their corresponding marker IDs
        for further calibration or tracking purposes.

        :param img_list: List of input images containing ArUco markers to be
            detected and processed
        :type img_list: list[np.ndarray]
        :return: Tuple containing two lists - the first list holds detected
            corner coordinates for each image, and the second list holds
            corresponding marker IDs for each image
        :rtype: tuple[list, list]
        """
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

    def calibrate(self, img_list: list[np.ndarray], save=True, filename=None):
        """
        Performs camera calibration using a list of images containing calibration patterns.

        This method processes multiple images to determine the camera's intrinsic parameters
        including the camera matrix and distortion coefficients. It detects calibration
        pattern corners in each image, matches them with object points, and runs OpenCV's
        camera calibration algorithm with specific flags to fix certain distortion
        parameters. The calibration results can optionally be saved to a file.

        :param img_list: List of images containing the calibration pattern, each image
            should be a numpy array with the same dimensions
        :type img_list: list[np.ndarray]
        :param save: Whether to save the camera matrix to a file, defaults to True
        :type save: bool
        :param filename: Path where the camera matrix should be saved. If None, generates
            a timestamped filename in the default config directory
        :type filename: str or Path or None
        :return: Tuple containing the calibration error (RMS re-projection error), camera
            matrix, distortion coefficients, rotation vectors, and translation vectors
        :rtype: tuple
        """
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
        if save:
            time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = Path(filename) if filename \
                else Path(__file__).parent / 'data' / 'config' / f'intrinsic-{time_now}.npy'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, mtx)

        return ret, mtx, dist, rvecs, tvecs

    def compute_turntable_transform(self, tvecs, height, radial, save=True, filename=None):
        """
        Compute transformation matrix from camera coordinates to turntable coordinates using SVD-based plane fitting and circle fitting.

        This method takes a set of translation vectors representing camera poses, fits a plane through them using Singular Value Decomposition, determines the circle center on that plane using least squares, and constructs a coordinate system aligned with the turntable geometry. The resulting transformation matrix converts points from camera coordinates to the turntable's natural coordinate system.

        :param tvecs: Collection of translation vectors from camera calibration
        :type tvecs: array-like
        :param height: Height parameter of the turntable configuration
        :type height: float
        :param radial: Radial parameter of the turntable configuration
        :type radial: float
        :param save: Whether to save the computed transformation to disk
        :type save: bool
        :param filename: Path where transformation data should be saved; if None, generates timestamp-based filename
        :type filename: str or None
        :return: 4x4 homogeneous transformation matrix from camera to turntable coordinates
        :rtype: numpy.ndarray
        """
        points = np.array([t.flatten() for t in tvecs])

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        U, S, Vt = np.linalg.svd(centered_points)
        normal = Vt[2, :]
        normal = normal if normal[-1] < 0 else -normal
        z_new = normal / np.linalg.norm(normal)

        x_new = np.cross(z_new, centroid)
        x_new = x_new / np.linalg.norm(x_new)

        y_new = np.cross(z_new, x_new)
        y_new = y_new / np.linalg.norm(y_new)

        w2c = np.eye(4)
        w2c[:3, 0], w2c[:3, 1], w2c[:3, 2], w2c[:3, 3] = x_new, y_new, z_new, centroid

        if save:
            save_dict = {
                'height': np.array(height),
                'radial': np.array(radial),
                'matrix': w2c
            }
            time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = Path(filename) if filename \
                else Path(__file__).parent / 'data' / 'config' / f'extrinsic-{time_now}.npy'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_path, **save_dict)
        return w2c





if __name__ == '__main__':
    cali = Calibrator()
    cali_base = Path(__file__).parent / 'data' / 'cali'
    img_paths = list(cali_base.rglob('*uint8.png'))
    all_imgs = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_paths]
    ret, mtx, dist, rvecs, tvecs = cali.calibrate(all_imgs, filename='data/config/intrinsic-rgbd.npy')
    cali.compute_turntable_transform(tvecs[:12], 0.07, 0.2, filename='data/config/extrinsic-rgbd.npz')
    img_paths = list(cali_base.rglob('*.tiff'))
    all_imgs = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_paths]
    ret, mtx, dist, rvecs, tvecs = cali.calibrate(all_imgs, filename='data/config/intrinsic-ms.npy')
    cali.compute_turntable_transform(tvecs[:12], 0.07, 0.2, filename='data/config/extrinsic-ms.npz')
