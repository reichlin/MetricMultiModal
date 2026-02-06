import gymnasium as gym
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from mujoco import mj_name2id, mjtObj  # from DeepMind mujoco
from mujoco import mj_forward


def depth_gl_to_meters(depth_gl, znear, zfar):
    # Convert OpenGL depth [0,1] to metric depth using near/far planes.
    depth_gl = np.clip(depth_gl, 1e-6, 1.0 - 1e-6)
    return (2.0 * znear * zfar) / (
        zfar + znear - (2.0 * depth_gl - 1.0) * (zfar - znear)
    )


def intrinsics_from_fovy(fovy_deg, W, H):
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * H / np.tan(0.5 * fovy)
    fx = fy * (W / H)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return fx, fy, cx, cy



class DepthObsWrapper(gym.ObservationWrapper):
    """
    Adds a single-channel depth image to the observation dict under key 'depth'.
    Tries the modern Gymnasium MuJoCo renderer first, then falls back to mujoco_py.
    """

    def __init__(self, env, width=128, height=128, camera_name=None, camera_id=-1):
        super().__init__(env)
        self.width, self.height = int(width), int(height)
        self.camera_name, self.camera_id = camera_name, int(camera_id)

        # If the base obs is a dict (goal-based), extend it; otherwise make it a dict.
        if isinstance(self.observation_space, gym.spaces.Dict):
            depth_space = gym.spaces.Box(
                low=0.0, high=np.inf, shape=(self.height, self.width), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(
                {
                    **self.observation_space.spaces,
                    "depth": depth_space,
                }
            )
        else:
            # Not typical for robotics goal envs, but handle classic control just in case.
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": self.observation_space,
                    "achieved_goal": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
                    ),
                    "desired_goal": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
                    ),
                    "depth": gym.spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(self.height, self.width),
                        dtype=np.float32,
                    ),
                }
            )

    def _render_rgb_depth(self):
        u = self.env.unwrapped

        # Always propagate model -> data before any rendering
        try:
            sid = mj_name2id(u.model, mjtObj.mjOBJ_SITE, "target0")
            bid = int(u.model.site_bodyid[sid])
            R_w_b = u.data.xmat[bid].reshape(3, 3)
            p_w_b = u.data.xpos[bid]
            goal_w = u.goal
            p_local = R_w_b.T @ (goal_w - p_w_b)
            u.model.site_pos[sid] = p_local
            mj_forward(u.model, u.data)
            # mj_forward(u.model, u.data)  # ensures site/geom poses (incl. goal) are up-to-date
        except Exception:
            # Fallback for older APIs that expose sim.forward()
            if hasattr(u, "sim") and hasattr(u.sim, "forward"):
                u.sim.forward()

        # Path A: Gymnasium MujocoRenderer
        if hasattr(u, "mujoco_renderer") and u.mujoco_renderer is not None:
            r = u.mujoco_renderer

            # (optional) if available, refresh the scene explicitly
            if hasattr(r, "update_scene"):
                try:
                    r.update_scene(data=u.data)
                except TypeError:
                    r.update_scene(u.data)

            # Select camera if requested
            try:
                if self.camera_name is not None or self.camera_id != -1:
                    # Gymnasium exposes a convenience setter
                    r.set_camera(
                        camera_name=self.camera_name
                        if self.camera_name is not None
                        else None,
                        camera_id=None if self.camera_id == -1 else self.camera_id,
                    )
            except TypeError:
                # Older versions may only accept one of the arguments;
                # try the available ones without breaking.
                if self.camera_name is not None:
                    r.set_camera(camera_name=self.camera_name)
                elif self.camera_id != -1:
                    r.set_camera(camera_id=self.camera_id)

            # Now render without kwargs
            rgb = r.render("rgb_array")
            depth = r.render("depth_array")
            return rgb, depth

        # Path B: mujoco_py-style API (legacy)
        if hasattr(u, "sim") and hasattr(u.sim, "render"):
            out = u.sim.render(
                width=self.width,
                height=self.height,
                camera_name=self.camera_name,
                depth=True,
            )
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[0], out[1]

        raise RuntimeError("No compatible renderer found for RGB+Depth.")

    def observation(self, obs):
        rgb, depth = self._render_rgb_depth()
        # Ensure expected shapes/dtypes
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)
        # Optionally, you could normalize here; we leave raw MuJoCo depth.
        if isinstance(obs, dict):
            obs["depth"] = depth
            obs["rgb"] = rgb
            return obs
        else:
            return {
                "observation": obs,
                "achieved_goal": np.array([], dtype=np.float32),
                "desired_goal": np.array([], dtype=np.float32),
                "depth": depth,
                "rgb": rgb,
            }


class PointCloudObsWrapper(gym.ObservationWrapper):
    """
    Requires that observations already contain:
      - 'depth': (H, W) float array (meters or OpenGL depth in [0,1])
      - 'rgb':   (H, W, 3) uint8
    Adds:
      - 'pointcloud': (N, 3) float32 (camera or world frame)
      - 'pc_rgb':     (N, 3) uint8    colors aligned with points
    """

    def __init__(
        self,
        env,
        camera_name=None,
        stride=1,
        world_frame=True,
        near_clip=0.03,
        far_clip=5.0,
    ):
        super().__init__(env)
        self.camera_name = camera_name
        self.stride = int(stride)
        self.world_frame = bool(world_frame)
        self.near_clip = float(near_clip)
        self.far_clip = float(far_clip)

        assert isinstance(self.observation_space, gym.spaces.Dict), (
            "PointCloudObsWrapper expects a Dict obs (wrap after DepthObsWrapper)."
        )
        # Variable-length outputs → declare 0-length shape in spaces
        self.observation_space.spaces["pointcloud"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(0, 3), dtype=np.float32
        )
        self.observation_space.spaces["pc_rgb"] = gym.spaces.Box(
            low=0, high=255, shape=(0, 3), dtype=np.uint8
        )

    # ---------- MuJoCo helpers (model/data) ----------
    def _cam_id(self, model):
        if self.camera_name is None:
            return 0
        cam_id = mj_name2id(model, mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in model.")
        return cam_id

    def _depth_to_meters(self, model, depth):
        # If depth looks like [0,1], convert from GL depth; otherwise assume meters.
        if np.isfinite(depth).any() and np.nanmax(depth) <= 1.0 + 1e-6:
            znear = float(model.vis.map.znear)
            zfar = float(model.vis.map.zfar)
            return depth_gl_to_meters(depth, znear, zfar)
        return depth

    def _camera_pose_world(self, data, cam_id):
        # World-from-camera rotation & translation
        R_w_c = data.cam_xmat[cam_id].reshape(3, 3).copy()
        t_w_c = data.cam_xpos[cam_id].copy()
        return R_w_c, t_w_c

    # ---------- Backprojection ----------
    def _backproject(self, depth_m, rgb, model, data, cam_id):
        H, W = depth_m.shape
        fx, fy, cx, cy = intrinsics_from_fovy(model.cam_fovy[cam_id], W, H)

        ys = np.arange(0, H-32, self.stride)
        xs = np.arange(0, W, self.stride)
        xv, yv = np.meshgrid(xs, ys)

        z = depth_m[yv, xv]  # (h, w)

        valid = np.isfinite(z) & (z > 0)
        valid &= (z > self.near_clip) & (z < self.far_clip)
        if not np.any(valid):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        u = xv[valid].astype(np.float32)
        v = yv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        Xc = np.stack([x, y, z], axis=1)  # (N, 3) camera frame
        C = rgb[yv[valid], xv[valid]].astype(np.uint8)

        if self.world_frame:
            R_w_c, t_w_c = self._camera_pose_world(data, cam_id)
            Xw = (R_w_c @ Xc.T).T + t_w_c[None, :]
            return Xw.astype(np.float32, copy=False), C
        else:
            return Xc.astype(np.float32, copy=False), C

    # ---------- Gym ObservationWrapper hook ----------
    def observation(self, obs):
        # Get model/data from the unwrapped env (modern Gymnasium robotics)
        u = self.env.unwrapped
        model = getattr(u, "model", None)
        data = getattr(u, "data", None)
        if model is None or data is None:
            raise RuntimeError(
                "Expected env.unwrapped to expose 'model' and 'data' (mujoco)."
            )

        depth = obs["depth"]
        rgb = obs["rgb"]

        cam_id = self._cam_id(model)
        depth_m = self._depth_to_meters(model, depth)
        pc, col = self._backproject(depth_m, rgb, model, data, cam_id)

        obs["pointcloud"] = pc
        obs["pc_rgb"] = col
        return obs








