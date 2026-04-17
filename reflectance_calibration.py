from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from scipy.spatial import KDTree


class MLPRegressor(torch.nn.Module):
    def __init__(self, in_dim=9, out_dim=25):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ReflectanceCalibrator:
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.mlp = MLPRegressor()
        self.is_fitted = False

    @staticmethod
    def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
        """PCA-based surface normal estimation (fully vectorized)."""
        tree = KDTree(points)
        _, idx = tree.query(points, k=k)
        neighbors = points[idx]  # (N, k, 3)
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)  # (N, k, 3)
        cov = np.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)
        _, eigvecs = np.linalg.eigh(cov)  # sorted ascending
        normals = eigvecs[:, :, 0]  # smallest eigenvalue
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        return normals

    @staticmethod
    def _orient_normals(normals: np.ndarray,
                        points: np.ndarray,
                        reference_point: np.ndarray) -> np.ndarray:
        """Flip normals so they consistently point toward *reference_point*."""
        flip = np.sum(normals * (reference_point - points), axis=1) < 0
        normals[flip] *= -1
        return normals

    # ------------------------------------------------------------------ #
    #  Coordinate transforms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _camera_origin_pre_ext(rgbd_extrinsic: np.ndarray) -> np.ndarray:
        """Camera origin [0,0,0] in the pre-ext_mtx frame."""
        return (np.linalg.inv(rgbd_extrinsic) @ np.array([0., 0., 0., 1.]))[:3]

    @staticmethod
    def _world_to_pre_ext(points_world: np.ndarray,
                          extra_mtx: np.ndarray) -> np.ndarray:
        """Inverse-transform world points back to the pre-ext_mtx frame."""
        pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
        return (np.linalg.inv(extra_mtx) @ pts_h.T).T[:, :3]

    # ------------------------------------------------------------------ #
    #  Feature construction
    # ------------------------------------------------------------------ #
    def _build_features(self, pts_pre: np.ndarray,
                        normals: np.ndarray,
                        cam_origin: np.ndarray) -> np.ndarray:
        """Concatenate [xyz_pre_ext | normal | view_dir] -> (N, 9)."""
        view_dirs = cam_origin - pts_pre
        view_dirs /= (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
        return np.hstack([pts_pre, normals, view_dirs])

    def _features_for_view(self, pc_world: np.ndarray,
                           extra_mtx: np.ndarray,
                           rgbd_extrinsic: np.ndarray) -> np.ndarray:
        cam_origin = self._camera_origin_pre_ext(rgbd_extrinsic)
        pts_pre = self._world_to_pre_ext(pc_world[:, :3], extra_mtx)
        normals = self.estimate_normals(pts_pre, k=self.k_neighbors)
        normals = self._orient_normals(normals, pts_pre, cam_origin)
        return self._build_features(pts_pre, normals, cam_origin)

    def fit(self,
            reference_pcs: list[np.ndarray],
            extra_mtxs: np.ndarray,
            extrinsic: np.ndarray,
            epochs: int = 10000,
            lr: float = 1e-3,
            batch_size: int = 1 << 16,
            test_size: float = 0.2,
            seed: int = 42,
            patience: int = 20,
            min_delta: float = 1e-6,
            verbose=False):
        all_x, all_y = [], []
        for i, pc in enumerate(reference_pcs):
            feat = self._features_for_view(pc, extra_mtxs[i], extrinsic)
            all_x.append(feat)
            all_y.append(pc[:, -25:])
        x = np.vstack(all_x)
        y = np.vstack(all_y)

        # ---------- train / test split ----------
        rng = np.random.default_rng(seed)
        n_total = len(x)
        perm = rng.permutation(n_total)
        n_test = int(n_total * test_size)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        if verbose:
            print(f"Dataset split  |  train = {len(x_train)}  test = {len(x_test)}")

        # ---------- prepare tensors & loader ----------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp.to(device)
        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

        # ---------- training loop ----------
        # ---------- training loop ----------
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        final_train_loss = 0.0
        final_test_loss = 0.0
        n_train = len(x_train_t)

        # ---------- early stopping state ----------
        best_test_loss = float('inf')
        best_state = {k: v.detach().clone() for k, v in self.mlp.state_dict().items()}
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.mlp.train()
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = x_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                pred = self.mlp(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            final_train_loss = epoch_loss / n_train

            # ---------- evaluate on test set ----------
            self.mlp.eval()
            with torch.no_grad():
                test_pred = self.mlp(x_test_t)
                final_test_loss = criterion(test_pred, y_test_t).item()
                # additional metrics: MAE and per-channel relative error
                test_mae = (test_pred - y_test_t).abs().mean().item()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch + 1:>4d}/{epochs}  "
                      f"train_loss = {final_train_loss:.6f}  "
                      f"test_loss = {final_test_loss:.6f}  "
                      f"test_mae = {test_mae:.6f}")

            # ---------- early stopping check ----------
            if final_test_loss < best_test_loss - min_delta:
                best_test_loss = final_test_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in self.mlp.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}  |  "
                              f"best epoch = {best_epoch}  "
                              f"best test_loss = {best_test_loss:.6f}")
                    break

        # ---------- restore best weights ----------
        self.mlp.load_state_dict(best_state)
        final_test_loss = best_test_loss

        # ---------- final test report ----------
        self.mlp.eval()
        with torch.no_grad():
            test_pred = self.mlp(x_test_t).cpu().numpy()
        y_test_np = y_test
        residual = test_pred - y_test_np
        ss_res = (residual ** 2).sum(axis=0)
        ss_tot = ((y_test_np - y_test_np.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
        r2_per_channel = 1.0 - ss_res / ss_tot
        if verbose:
            print(f"Test set R² (mean over channels) = {r2_per_channel.mean():.4f}")
            print(f"Test set R² (min/max channel)    = "
                  f"{r2_per_channel.min():.4f} / {r2_per_channel.max():.4f}")
            print(f"Training complete  |  "
                  f"final train loss = {final_train_loss:.6f}  |  "
                  f"final test loss  = {final_test_loss:.6f}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Inference & calibration
    # ------------------------------------------------------------------ #
    def predict_radiance(self,
                         pc_world: np.ndarray,
                         extra_mtx: np.ndarray,
                         rgbd_extrinsic: np.ndarray) -> np.ndarray:
        """Predict the expected radiance for each point in *pc_world*."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        feat = self._features_for_view(pc_world, extra_mtx, rgbd_extrinsic)

        device = next(self.mlp.parameters()).device
        x_t = torch.tensor(feat, dtype=torch.float32, device=device)

        self.mlp.eval()
        with torch.no_grad():
            y = self.mlp(x_t).cpu().numpy()

        return y

    def calibrate(self,
                  pc_world: np.ndarray,
                  observed_radiance: np.ndarray,
                  extra_mtx: np.ndarray,
                  rgbd_extrinsic: np.ndarray,
                  reference_reflectance: float = 1.0) -> np.ndarray:
        """
        Compute calibrated reflectance.

            reflectance = reference_reflectance × (observed / predicted)

        Parameters
        ----------
        pc_world : (N, 3+) array
            Target point cloud in world frame.
        observed_radiance : (N,) array
            Raw radiance measurements.
        extra_mtx : (4, 4) array
            Extra transform for this view.
        rgbd_extrinsic : (4, 4) array
            RGBD extrinsic matrix.
        reference_reflectance : float
            Known reflectance of the reference body (default 1.0 = 100 %).

        Returns
        -------
        (N,) array of calibrated reflectance values.
        """
        predicted = self.predict_radiance(pc_world, extra_mtx, rgbd_extrinsic)
        predicted = np.clip(predicted, 1e-6, None)
        observed = np.asarray(observed_radiance).ravel()
        return reference_reflectance * observed / predicted

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        joblib.dump({
            'mlp_state': self.mlp.state_dict(),
            'k_neighbors': self.k_neighbors,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.mlp.load_state_dict(data['mlp_state'])
        self.k_neighbors = data['k_neighbors']
        self.is_fitted = True
        return self


if __name__ == '__main__':
    from registration import Registrator

    registrator = Registrator()
    ref_base = Path('data/60')

    extra_mtx = registrator.get_extra_transform(list(ref_base.glob('*_v*.xlsx'))[0])
    num_views = extra_mtx.shape[0]

    rgb, depth, ms = [], [], []
    for i in range(num_views):
        d = ref_base / str(i + 1)
        rgb.append(np.array(Image.open(d / f'{i + 1}_color_uint8.png')))
        depth.append(np.array(Image.open(d / f'{i + 1}_depth_uint16.png')))
        ms.append(np.array(Image.open(d / f'{i + 1}.tiff')))

    rgb_np, depth_np, ms_np = np.array(rgb), np.array(depth), np.array(ms)

    # --- build reference point clouds with MS radiance ---
    rgb_pcs = registrator.get_rgb_pc(rgb_np, depth_np, extra_mtx, offset=0.025, save=False)
    ms_pcs = registrator.get_ms_pc(rgb_pcs, ms_np, extra_mtx, save=False)

    calibrator = ReflectanceCalibrator(k_neighbors=20)
    calibrator.fit(ms_pcs, extra_mtx, registrator.rgbd_extrinsic, verbose=True)
    calibrator.save('data/config/reflectance_calibrator.pkl')

    # # --- calibrate a new scene ---
    # scene_base = Path('data/60')
    # scene_extra = registrator.get_extra_transform(list(scene_base.glob('*_v*.xlsx'))[0])
    # # ... load scene rgb/depth/ms the same way ...
    # # scene_rgb_pcs = registrator.get_rgb_pc(...)
    # # scene_ms_pcs  = registrator.get_ms_pc(...)
    #
    # # for i, pc in enumerate(scene_ms_pcs):
    # #     reflectance = calibrator.calibrate(
    # #         pc[:, :3], pc[:, -1],
    # #         scene_extra[i], registrator.rgbd_extrinsic,
    # #         reference_reflectance=0.99,  # known reflectance of the reference panel
    # #     )
