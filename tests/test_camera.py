"""Milestone 3 gate: the PinholeCamera module, standalone, before it's
wired into scene.py/arm_mesh.py (Milestone 4). Locks in the calibrated
default pose (camera.default_pose) as a real regression test, not just a
throwaway scratchpad measurement.
"""
import numpy as np
import pytest

from arm import SO101PlanarArm
from camera import PinholeCamera, default_pose


@pytest.fixture
def arm():
    return SO101PlanarArm(include_shoulder_pan=True)


@pytest.fixture
def workspace_sample(arm):
    """Every link origin over a random joint-limit sweep, plus all 8
    target-box corners -- the same sample cloud calibrate_camera.py used."""
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(1500):
        q = rng.uniform(arm._lo, arm._hi)
        for name, (R, p) in arm.link_transforms(q).items():
            pts.append(p)
    for xs in (arm.target_lo[0], arm.target_hi[0]):
        for ys in (arm.target_lo[1], arm.target_hi[1]):
            for zs in (arm.target_lo[2], arm.target_hi[2]):
                pts.append(np.array([xs, ys, zs]))
    return np.array(pts)


def test_project_single_point_roundtrips_shape(arm):
    cam = PinholeCamera(image_size=(160, 160), **default_pose(arm))
    px, depth = cam.project(arm.fk(np.zeros(3)))
    assert px.shape == (2,)
    assert isinstance(depth, float)


def test_project_batch_shape(arm):
    cam = PinholeCamera(image_size=(160, 160), **default_pose(arm))
    pts = np.zeros((5, 3))
    px, depth = cam.project(pts)
    assert px.shape == (5, 2)
    assert depth.shape == (5,)


def test_forward_point_projects_near_center(arm):
    """A point exactly at look_at must project to (near) the principal
    point -- a basic sanity check on the projection math's sign/handedness."""
    kw = default_pose(arm)
    cam = PinholeCamera(image_size=(160, 160), **kw)
    px, depth = cam.project(kw["look_at"])
    assert px == pytest.approx([cam.cx, cam.cy], abs=1e-6)
    assert depth > 0


def test_up_point_moves_to_lower_row(arm):
    """A world point directly ABOVE look_at (same x,y, +z) must project to
    a SMALLER row (py) -- 'row grows downward' convention, matching
    scene.py's existing to_pixels()."""
    kw = default_pose(arm)
    cam = PinholeCamera(image_size=(160, 160), **kw)
    above = kw["look_at"] + np.array([0.0, 0.0, 0.1])
    px_center, _ = cam.project(kw["look_at"])
    px_above, _ = cam.project(above)
    assert px_above[1] < px_center[1]


def test_default_pose_covers_full_workspace(arm, workspace_sample):
    """The calibrated pose (camera.default_pose) must keep effectively the
    ENTIRE measured workspace sample in-frame with margin and in front of
    the camera -- this is the actual calibration gate, not just a
    scratchpad print (calibrate_camera.py, scratchpad, produced this
    pose). Threshold 99.5%, not 100%: the pose was moved closer
    for a bigger apparent silhouette, and re-measured at
    99.94% coverage (5px margin) / 99.9% (8px margin, this test's own
    margin) -- an accepted, deliberate tradeoff, not a regression."""
    cam = PinholeCamera(image_size=(160, 160), **default_pose(arm))
    ok = cam.in_frame(workspace_sample, margin_px=8.0, min_depth=0.05)
    assert ok.mean() >= 0.995, f"only {ok.mean():.1%} of the workspace sample stayed in-frame"


def test_default_pose_fills_a_reasonable_fraction_of_the_frame(arm, workspace_sample):
    """Not just 'in frame' -- the workspace shouldn't be a tiny speck in
    the middle of 160x160 either (a real, useful camera view)."""
    cam = PinholeCamera(image_size=(160, 160), **default_pose(arm))
    pixels, depth = cam.project(workspace_sample)
    visible = depth > 0.05
    px = pixels[visible]
    width_used = px[:, 0].max() - px[:, 0].min()
    height_used = px[:, 1].max() - px[:, 1].min()
    assert width_used > 100, f"only {width_used:.0f}px of 160 used horizontally"
    assert height_used > 80, f"only {height_used:.0f}px of 160 used vertically"


def test_behind_camera_point_has_negative_depth():
    cam = PinholeCamera(position=[0, 0, 0], look_at=[1, 0, 0], fov_deg=60, image_size=(160, 160))
    px, depth = cam.project([-1.0, 0.0, 0.0])
    assert depth < 0


def test_unproject_round_trip(arm):
    """project() then unproject_ray() must recover the SAME ray -- walking
    the unit ray by t=depth/dot(ray,forward) (NOT by `depth` directly,
    since unproject_ray returns a unit vector, not the raw unnormalized
    ray project() implicitly uses) must reproduce the original point."""
    cam = PinholeCamera(image_size=(160, 160), **default_pose(arm))
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform([-0.3, -0.3, -0.1], [0.5, 0.3, 0.5])
        px, depth = cam.project(p)
        if depth <= 0:
            continue
        ray = cam.unproject_ray(px)
        t = depth / np.dot(ray, cam.forward)
        recovered = cam.position + t * ray
        assert recovered == pytest.approx(p, abs=1e-9)
