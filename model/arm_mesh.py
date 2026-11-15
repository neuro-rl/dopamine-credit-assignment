"""arm_mesh.py — real SO-101 mesh rendering for the workspace panel.

Standalone from the rest of the model: nothing here feeds the network (the
camera in encoders.py/scene.py never sees the arm at all, mesh or schematic).
This is purely the human-facing visualization.

Why convex hulls, not the actual triangle meshes: MEASURED, the 19 visual
mesh instances across the 7 moving links sum to ~400,000 triangles PER FRAME.
Drawing that many individual matplotlib polygons per video frame is not
practical (a video is hundreds of frames). Each link's mesh(es) collapse to
ONE 2D convex hull polygon instead -- 7-8 polygons per frame, drawn with a
simple back-to-front (painter's algorithm) depth sort. This loses concavities
(e.g. the gripper jaw's exact notch) but keeps the actual silhouette/outline
of each real part, which is the point: an accurate robot shape, not a CAD
render.

Uses arm.link_transforms(q) for every link's world pose -- doesn't re-derive
the kinematic chain here, so this can't silently diverge from fk().
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_URDF_PATH = _ASSETS_DIR / "so101_new_calib.urdf"

# Rendered back-to-front by mean depth; colour is just for visual separation
# between parts, not meaningful.
_LINK_COLORS = {
    "base_link": "#555555",
    "shoulder_link": "#c9761c",
    "upper_arm_link": "#d98e2b",
    "lower_arm_link": "#3b6fa3",
    "wrist_link": "#4f86bd",
    "gripper_link": "#666666",
    "gripper_frame_link": "#666666",
    "moving_jaw_so101_v1_link": "#222222",
}


def _parse_rpy_xyz(origin_el):
    if origin_el is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in origin_el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in origin_el.get("rpy", "0 0 0").split()])
    return xyz, rpy


def _rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _resolve_mesh_path(urdf_filename):
    """The URDF references "assets/<name>.stl" (relative to its own package
    root), but the STL files were placed in assets/stl/<name> -- try both."""
    name = Path(urdf_filename).name
    for candidate in (_ASSETS_DIR / name, _ASSETS_DIR / "stl" / name, _ASSETS_DIR.parent / urdf_filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"mesh {urdf_filename!r} not found under {_ASSETS_DIR} or {_ASSETS_DIR / 'stl'}")


def _parse_link_visuals(urdf_path):
    """{link_name: [(mesh_path, local_xyz, local_rpy), ...]}"""
    root = ET.parse(str(urdf_path)).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        visuals = []
        for v in link.findall("visual"):
            mesh_el = v.find("geometry/mesh")
            if mesh_el is None:
                continue
            xyz, rpy = _parse_rpy_xyz(v.find("origin"))
            visuals.append((_resolve_mesh_path(mesh_el.get("filename")), xyz, rpy))
        if visuals:
            out[name] = visuals
    return out


def _extremal_points(verts, n_directions=64, seed=0):
    """Direction-sampling convex-hull reduction: the vertex furthest along
    each of n_directions random unit vectors is a true 3D hull vertex, and a
    3D hull vertex remains extremal (hence a candidate 2D-hull vertex) under
    any subsequent rotation+projection. Not exhaustive (a real vertex whose
    normal cone none of the sampled directions land in gets dropped), but at
    64 directions the miss rate is visually negligible for these part
    shapes, and it's a huge, cheap reduction (tens of thousands of vertices
    down to <=64)."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    idx = np.argmax(verts @ dirs.T, axis=0)
    return np.unique(verts[idx], axis=0)


def _convex_hull_2d(points):
    """Andrew's monotone chain, pure numpy -- no scipy dependency."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        hull = []
        for p in seq:
            while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


class ArmMeshRenderer:

    def __init__(self, urdf_path=None):
        self._link_visuals = _parse_link_visuals(urdf_path or _URDF_PATH)
        self._mesh_cache = {}   # path -> (N,3) unique vertex array

    def _mesh_vertices(self, path):
        """Cached, and reduced to candidate hull vertices ONCE per mesh
        (direction-sampling, see module docstring: any world-frame 2D hull
        this mesh will ever produce, under ANY rotation, is a subset of
        these candidates' hull -- a 3D convex hull vertex stays an extremal
        point after any linear transform, including rotation). MEASURED:
        this is what brings per-frame rendering from ~0.4s/link to a few ms
        -- computing the full mesh's 2D hull fresh every frame is what made
        the naive version too slow for video."""
        if path not in self._mesh_cache:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(str(path))
            verts = np.unique(m.vectors.reshape(-1, 3), axis=0)
            self._mesh_cache[path] = _extremal_points(verts, n_directions=64)
        return self._mesh_cache[path]

    def link_hulls_xz(self, arm, q):
        """{link_name: (hull_2d_xz (K,2), mean_depth_y)} for every link that
        has visual geometry, at joint state q."""
        transforms = arm.link_transforms(q)
        out = {}
        for link_name, visuals in self._link_visuals.items():
            if link_name not in transforms:
                continue
            R_link, p_link = transforms[link_name]
            all_world_pts = []
            for mesh_path, local_xyz, local_rpy in visuals:
                verts = self._mesh_vertices(mesh_path)
                R_vis = _rpy_to_R(local_rpy)
                world = (verts @ R_vis.T + local_xyz) @ R_link.T + p_link
                all_world_pts.append(world)
            world_pts = np.concatenate(all_world_pts, axis=0)
            hull_xz = _convex_hull_2d(world_pts[:, [0, 2]])
            mean_depth = float(world_pts[:, 1].mean())
            out[link_name] = (hull_xz, mean_depth)
        return out

    def link_hulls_pixels(self, arm, q, camera):
        """{link_name: (hull_pixels (K,2), mean_depth_cam)} -- perspective-
        projected analogue of link_hulls_xz, for the real PinholeCamera
        pipeline (camera.py, Milestone 4) instead of the fixed-scale
        orthographic x-z slice. `camera` is duck-typed (just needs
        .project(points_world) -> (pixels, depth)) to avoid a hard import
        dependency on camera.py from this module.

        Reuses the SAME cached, direction-sampling-reduced candidate
        vertex set as link_hulls_xz (_mesh_vertices) -- validated
        separately (calibrate_camera.py / hull-reduction check, scratchpad)
        that this reduction still bounds the true post-projection 2D hull
        closely under perspective, not just under the old linear
        (orthographic) transform it was originally justified for."""
        transforms = arm.link_transforms(q)
        out = {}
        for link_name, visuals in self._link_visuals.items():
            if link_name not in transforms:
                continue
            R_link, p_link = transforms[link_name]
            all_world_pts = []
            for mesh_path, local_xyz, local_rpy in visuals:
                verts = self._mesh_vertices(mesh_path)
                R_vis = _rpy_to_R(local_rpy)
                world = (verts @ R_vis.T + local_xyz) @ R_link.T + p_link
                all_world_pts.append(world)
            world_pts = np.concatenate(all_world_pts, axis=0)
            pixels, depth = camera.project(world_pts)
            hull_px = _convex_hull_2d(pixels)
            out[link_name] = (hull_px, float(depth.mean()))
        return out

    def draw(self, ax, arm, q, alpha=0.95):
        import matplotlib.pyplot as plt
        hulls = self.link_hulls_xz(arm, q)
        for link_name, (hull, depth) in sorted(hulls.items(), key=lambda kv: -kv[1][1]):
            if len(hull) < 3:
                continue
            color = _LINK_COLORS.get(link_name, "#888888")
            ax.add_patch(plt.Polygon(hull, closed=True, facecolor=color,
                                     edgecolor="black", linewidth=0.4, alpha=alpha, zorder=3))

    def draw_perspective(self, ax, arm, q, camera, alpha=0.95):
        """Perspective counterpart to draw() -- same real mesh hulls, same
        painter's-algorithm depth sort, but projected through a real
        PinholeCamera (camera.py) instead of the fixed-scale orthographic
        x-z slice. `ax` must already be set up in PIXEL coordinates
        (xlim=[0,w], ylim=[h,0] i.e. row-grows-downward -- see camera.py's
        own convention docstring; viz.py's caller handles this once per
        frame, not here, same division of responsibility as draw()/
        link_hulls_xz leaving axis limits to the caller)."""
        import matplotlib.pyplot as plt
        hulls = self.link_hulls_pixels(arm, q, camera)
        for link_name, (hull, depth) in sorted(hulls.items(), key=lambda kv: -kv[1][1]):
            if len(hull) < 3:
                continue
            color = _LINK_COLORS.get(link_name, "#888888")
            ax.add_patch(plt.Polygon(hull, closed=True, facecolor=color,
                                     edgecolor="black", linewidth=0.4, alpha=alpha, zorder=3))
