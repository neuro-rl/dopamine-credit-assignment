"""camera.py -- a real pinhole/perspective camera model.

Standalone, unit-validated here, NOT yet wired into scene.py/arm_mesh.py
(that's Milestone 4) -- this module only defines the projection math and a
calibration helper, deliberately isolated from any NEST/training-path code
so a geometry bug can be caught and fixed here, cheaply, before it can
propagate into the render pipeline.

Convention (matches scene.py's existing to_pixels()'s own "row grows
downward" convention, so nothing downstream needs a handedness change):
  - World frame: arm.py's own (x, y, z), z "up".
  - Camera frame: +Z_cam = forward (into the scene), +X_cam = right,
    +Y_cam = up (camera-local, NOT image-row direction).
  - Image: column (px) grows with +X_cam; row (py) grows DOWNWARD, i.e.
    opposite +Y_cam -- the standard image-plane convention.

There is no other perspective-camera code anywhere in this repo (confirmed
by exhaustive grep during planning) -- everything visual before this was a
fixed-scale orthographic affine map (scene.py's to_pixels(), arm_mesh.py's
link_hulls_xz literally slicing out world x/z and dropping y). This is the
real thing, meant to serve the project's actual long-term goal (real-
camera/real-robot transfer), not a placeholder.
"""
import numpy as np


def default_pose(arm):
    """A calibrated (not guessed) camera pose for a given 3-DOF arm.

    Moved closer than an earlier pose, to avoid the pool-activity
    panel looking like a blob rather than a recognizable silhouette:
    0.65m back / 0.26m up (was 1.0m / 0.4m), fov_deg widened 50->65 to
    compensate for being closer (re-swept, calibrate_camera.py,
    scratchpad, 4000-sample joint-limit sweep + 8 target-box corners):
    frac_in_frame=99.94% at 5px margin (vs the old pose's 100.0% at 8px
    margin -- a small, accepted coverage tradeoff for a real net gain in
    apparent size). Net apparent-size gain is NOT simply "35% closer" --
    the wider FOV needed for coverage partially cancels it: f_px/distance
    (the actual size-determining ratio) went from 102.9/1.0=102.9 to
    75.3/0.65=115.8, a real but modest ~12.5% increase, not the naive
    ~54% the distance change alone would suggest. (The blob look itself
    was separately diagnosed as mostly an accumulated-spike-count-over-
    the-fixation-window effect, not a resolution/size problem per se --
    see diag_pool_silhouette.py, scratchpad -- so don't expect this
    alone to make pool activity crisply silhouette-shaped.)

    Returns kwargs for PinholeCamera(**kwargs)."""
    reach_center = np.array([arm.pan_x0, arm.pan_y0, arm.pivot_xyz(np.zeros(3))[2]])
    position = reach_center + np.array([-0.65, 0.0, 0.26])
    return dict(position=position, look_at=reach_center, fov_deg=65.0)


class PinholeCamera:
    def __init__(self, position, look_at, fov_deg=60.0, image_size=(160, 160),
                 world_up=(0.0, 0.0, 1.0)):
        self.position = np.asarray(position, dtype=float)
        self.look_at = np.asarray(look_at, dtype=float)
        self.fov_deg = float(fov_deg)
        self.image_size = tuple(image_size)

        forward = self.look_at - self.position
        forward_norm = np.linalg.norm(forward)
        assert forward_norm > 1e-9, "camera position and look_at coincide"
        self.forward = forward / forward_norm

        up_hint = np.asarray(world_up, dtype=float)
        right = np.cross(self.forward, up_hint)
        right_norm = np.linalg.norm(right)
        assert right_norm > 1e-9, (
            "camera forward direction is parallel to world_up -- pick a "
            "different world_up hint or camera pose")
        self.right = right / right_norm
        self.true_up = np.cross(self.right, self.forward)

        h, w = self.image_size
        self.cx = (w - 1) / 2.0
        self.cy = (h - 1) / 2.0
        self.f_px = 0.5 * w / np.tan(np.radians(self.fov_deg) / 2.0)

    def project(self, points_world):
        """World (N,3) or (3,) -> (pixels (N,2) or (2,), depth_cam (N,) or
        scalar). depth_cam is the true camera-space z (distance along
        `forward`, signed) -- positive means in front of the camera,
        negative/zero means behind or at the optical center (caller should
        treat those as not-visible, NOT feed them through the perspective
        divide and trust the result -- see calibrate.py's own margin
        check)."""
        pts = np.asarray(points_world, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        rel = pts - self.position
        x_cam = rel @ self.right
        y_cam = rel @ self.true_up
        z_cam = rel @ self.forward

        eps = 1e-9
        z_safe = np.where(np.abs(z_cam) < eps, np.sign(z_cam) * eps + eps, z_cam)
        u = x_cam / z_safe
        v = y_cam / z_safe
        px = self.cx + self.f_px * u
        py = self.cy - self.f_px * v
        pixels = np.stack([px, py], axis=-1)

        if single:
            return pixels[0], float(z_cam[0])
        return pixels, z_cam

    def in_frame(self, points_world, margin_px=0.0, min_depth=0.01):
        """Convenience check used by the calibration script: True where a
        world point projects inside the image bounds (with `margin_px`
        slack on every side) AND is genuinely in front of the camera
        (depth > min_depth, not just nonzero)."""
        pixels, depth = self.project(points_world)
        pixels = np.atleast_2d(pixels)
        depth = np.atleast_1d(depth)
        h, w = self.image_size
        in_x = (pixels[:, 0] >= -margin_px) & (pixels[:, 0] < w + margin_px)
        in_y = (pixels[:, 1] >= -margin_px) & (pixels[:, 1] < h + margin_px)
        in_front = depth > min_depth
        return in_x & in_y & in_front

    def unproject_ray(self, pixel):
        """Inverse of project(): pixel (2,) or (N,2) -> unit-length world-
        space ray direction (3,) or (N,3) from THIS camera's optical
        center through that pixel. project(self.position + t*ray) for any
        t>0 reproduces `pixel` (self-checked in tests/test_camera.py) --
        the exact algebraic inverse of the perspective-divide + intrinsics
        step in project()."""
        px = np.asarray(pixel, dtype=float)
        single = px.ndim == 1
        px = np.atleast_2d(px)
        u = (px[:, 0] - self.cx) / self.f_px
        v = -(px[:, 1] - self.cy) / self.f_px
        ray = (self.forward[None, :] + u[:, None] * self.right[None, :]
               + v[:, None] * self.true_up[None, :])
        ray = ray / np.linalg.norm(ray, axis=1, keepdims=True)
        return ray[0] if single else ray


