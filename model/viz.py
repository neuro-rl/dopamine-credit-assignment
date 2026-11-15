"""viz.py — episode video + summary plots.

Panels (horizontal): workspace (always), camera / events / pool (mode B
only -- blank + "mode A" text otherwise). The video must make "did the arm
reach it or not" answerable at a glance -- that's its whole purpose.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers projection="3d"
import numpy as np
import cv2

from arm import SO101PlanarArm

# Readability constants for the video/frame render -- default matplotlib
# sizes are illegible once a frame is downscaled into a print figure.
_TITLE_FONTSIZE = 16
_TICK_FONTSIZE = 12
_LEGEND_FONTSIZE = 13
_SUPTITLE_FONTSIZE = 18

# Lazy, cached: only try once. Falls back to the line-schematic if the STL
# assets or numpy-stl aren't present -- this project's URDF-only assets
# don't require the mesh files, so the tool must still work without them.
_mesh_renderer = None
_mesh_renderer_unavailable = False


def _get_mesh_renderer():
    global _mesh_renderer, _mesh_renderer_unavailable
    if _mesh_renderer_unavailable:
        return None
    if _mesh_renderer is None:
        try:
            from arm_mesh import ArmMeshRenderer
            _mesh_renderer = ArmMeshRenderer()
        except Exception:
            _mesh_renderer_unavailable = True
            return None
    return _mesh_renderer


# 2-DOF perspective workspace camera: a real PinholeCamera
# (camera.py) view of the real mesh (arm_mesh.py's draw_perspective),
# replacing the old fixed-scale orthographic x-z slice. camera.py's own
# default_pose() assumes include_shoulder_pan=True (reads arm.pan_x0/
# pan_y0, which only exist on a 3-DOF arm instance -- AttributeError on
# a 2-DOF arm) -- this is a separate pose, calibrated directly for the
# 2-DOF case: offset [0, -0.7, 0.30] from the workspace
# centre (arm.pivot, at the arm's own fixed motion-plane Y), fov=70deg,
# MEASURED (300 random joint configs + the full target_lo/target_hi
# box corners) to give 100% in-frame coverage of both the arm's full
# range of motion and the target box, vs. e.g. the naive edge-on choice
# of offsetting only in (x,z) like default_pose's own 3-DOF offset
# (which points the camera INTO the arm's own motion plane -- the arm
# only ever appears as a near-degenerate sliver, since a 2-DOF arm's
# motion is entirely confined to one Y=const plane). Y-offset-dominant
# instead gives a near-frontal, genuinely 3D (real mesh, real
# perspective, real depth) view of that plane from outside it.
_workspace_camera = None


def _get_workspace_camera(arm):
    global _workspace_camera
    if _workspace_camera is None:
        from camera import PinholeCamera
        y0 = float(arm.pivot_xyz(np.zeros(2))[1])
        reach_center = np.array([arm.pivot[0], y0, arm.pivot[1]])
        _workspace_camera = PinholeCamera(
            position=reach_center + np.array([0.0, -0.7, 0.30]),
            look_at=reach_center, fov_deg=70.0, image_size=(240, 240))
    return _workspace_camera


def _project_loop(camera, xz_points, y0):
    """World (x,z) points (at the arm's fixed motion-plane Y) -> pixel-
    space (N,2), for drawing target markers/guide circles/trail through
    the SAME real camera the mesh itself is projected through -- so
    everything in the panel (arm, target, success radius, reach-envelope
    guides, trail) lives in one consistent pixel frame, not a mix of
    world-space and ad-hoc screen-space conventions."""
    xz = np.atleast_2d(np.asarray(xz_points, dtype=float))
    world = np.column_stack([xz[:, 0], np.full(len(xz), y0), xz[:, 1]])
    pixels, _ = camera.project(world)
    return pixels


def _draw_schematic_arm(ax, arm: SO101PlanarArm, q):
    """Fallback when the real STL meshes aren't available: a schematic, not
    a CAD render, but still distinguishes it from a bare two-segment
    pendulum (Acrobot's usual look) with a mounted base, two differently-
    coloured links to scale, and a gripper marker at the tip instead of a
    plain dot.

    Draws exactly what arm.fk()/arm.elbow_xz() compute -- no separate
    hand-derived link geometry here: arm.py and the renderer must call the
    same FK, or the video shows the target in the wrong place and looks
    fine doing it.
    """
    p_base = np.array([0.0, 0.0])     # true base_link origin
    p0 = arm.pivot                    # shoulder_lift pivot (shoulder_pan locked at 0 -- fixed offset from base)
    p1 = arm.elbow_xz(q)
    p2 = arm.fk(q)

    # Mounted base plinth (SO-101 bolts to a desk -- it isn't a pendulum
    # pivoting from a free point).
    base = plt.Rectangle((p_base[0] - 0.018, p_base[1] - 0.014), 0.036, 0.028,
                         color="dimgray", zorder=2, ec="black", linewidth=0.5)
    ax.add_patch(base)

    # Fixed pedestal from the true base to the shoulder_lift pivot (base_link
    # -> shoulder_pan mount -> shoulder_lift housing; shoulder_pan is locked
    # at 0, so this segment never moves) -- then the two ACTIVE links, to
    # scale, in distinct colours (shoulder_lift / elbow_flex).
    ax.plot([p_base[0], p0[0]], [p_base[1], p0[1]], "-",
            color="darkslategray", linewidth=5, solid_capstyle="round", zorder=3)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-",
            color="tab:orange", linewidth=7, solid_capstyle="round", zorder=3)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
            color="tab:blue", linewidth=6, solid_capstyle="round", zorder=3)

    # Joints.
    for p, r in ((p0, 0.010), (p1, 0.008)):
        ax.add_patch(plt.Circle(p, r, color="black", zorder=4))

    # Gripper: two short prongs splayed off the end of link 2, instead of a
    # bare point -- SO-101 ends in a jaw, not a pendulum tip.
    link_dir = (p2 - p1)
    link_dir = link_dir / (np.linalg.norm(link_dir) + 1e-9)
    prong_len = 0.018
    for angle_deg in (18, -18):
        a = np.radians(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        prong_end = p2 + rot @ link_dir * prong_len
        ax.plot([p2[0], prong_end[0]], [p2[1], prong_end[1]], "-",
                color="black", linewidth=3, solid_capstyle="round", zorder=4)


def _draw_workspace_perspective(ax, arm, q, target, trail, success_radius, d_mm, mesh_renderer):
    """Real mesh, real PinholeCamera perspective -- see _get_workspace_
    camera's own comment for the calibrated pose. Everything in the panel
    (arm, target, success radius, reach-envelope guides, trail) is
    projected through the SAME camera into one consistent pixel frame."""
    camera = _get_workspace_camera(arm)
    y0 = float(arm.pivot_xyz(np.zeros(2))[1])
    w, h = camera.image_size
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # row grows downward -- camera.py's own convention
    ax.set_aspect("equal")
    ax.set_title(f"workspace  d={d_mm:.1f}mm", fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)

    theta = np.linspace(0, 2 * np.pi, 64)
    for r in (arm.MIN_REACH, arm.MAX_REACH):
        loop_xz = np.column_stack([arm.pivot[0] + r * np.cos(theta),
                                    arm.pivot[1] + r * np.sin(theta)])
        loop_px = _project_loop(camera, loop_xz, y0)
        ax.plot(loop_px[:, 0], loop_px[:, 1], "--", color="gray", linewidth=0.7)

    mesh_renderer.draw_perspective(ax, arm, q, camera)

    # Target + success radius + trail, projected through the same camera,
    # drawn on top of the mesh.
    circle_xz = np.column_stack([target[0] + success_radius * np.cos(theta),
                                  target[1] + success_radius * np.sin(theta)])
    circle_px = _project_loop(camera, circle_xz, y0)
    ax.plot(circle_px[:, 0], circle_px[:, 1], "-", color="tab:green", linewidth=1.5, zorder=5)
    target_px = _project_loop(camera, [target], y0)[0]
    ax.plot(target_px[0], target_px[1], "x", color="tab:red", markersize=8, markeredgewidth=2, zorder=5)

    if len(trail) > 1:
        trail_px = _project_loop(camera, np.asarray(trail), y0)
        n = len(trail_px)
        for i in range(n - 1):
            alpha = 0.15 + 0.85 * (i / max(n - 2, 1))
            ax.plot(trail_px[i:i + 2, 0], trail_px[i:i + 2, 1], "-",
                    color="tab:blue", alpha=alpha, linewidth=1.5, zorder=5)


def _draw_workspace_orthographic(ax, arm: SO101PlanarArm, q, target, trail, success_radius, d_mm, mesh_renderer):
    """The original fixed-scale orthographic x-z slice -- kept as the
    fallback when the real mesh/camera path is unavailable or fails."""
    x_lo = min(0.0, arm.pivot[0] - arm.MAX_REACH) - 0.05
    x_hi = arm.pivot[0] + arm.MAX_REACH + 0.05
    z_lo = min(0.0, arm.pivot[1] - arm.MAX_REACH) - 0.05
    z_hi = arm.pivot[1] + arm.MAX_REACH + 0.05
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_aspect("equal")
    ax.set_title(f"workspace  d={d_mm:.1f}mm", fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(arm.pivot[0] + arm.MIN_REACH * np.cos(theta),
            arm.pivot[1] + arm.MIN_REACH * np.sin(theta),
            "--", color="gray", linewidth=0.7)
    ax.plot(arm.pivot[0] + arm.MAX_REACH * np.cos(theta),
            arm.pivot[1] + arm.MAX_REACH * np.sin(theta),
            "--", color="gray", linewidth=0.7)

    if mesh_renderer is not None:
        mesh_renderer.draw(ax, arm, q)
    else:
        _draw_schematic_arm(ax, arm, q)

    # Target + success radius + trail, drawn on top of either renderer.
    circle = plt.Circle(target, success_radius, color="tab:green", fill=False, linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.plot(target[0], target[1], "x", color="tab:red", markersize=8, markeredgewidth=2, zorder=5)

    if len(trail) > 1:
        trail_arr = np.asarray(trail)
        n = len(trail_arr)
        for i in range(n - 1):
            alpha = 0.15 + 0.85 * (i / max(n - 2, 1))
            ax.plot(trail_arr[i:i + 2, 0], trail_arr[i:i + 2, 1], "-",
                    color="tab:blue", alpha=alpha, linewidth=1.5, zorder=5)


_perspective_workspace_unavailable = False


def _draw_workspace(ax, arm: SO101PlanarArm, q, target, trail, success_radius, d_mm):
    global _perspective_workspace_unavailable
    ax.clear()
    mesh_renderer = _get_mesh_renderer()
    if mesh_renderer is not None and not _perspective_workspace_unavailable:
        try:
            _draw_workspace_perspective(ax, arm, q, target, trail, success_radius, d_mm, mesh_renderer)
            return
        except Exception:
            # Never let a video/viz failure take down a training run --
            # same non-fatal philosophy as _get_mesh_renderer's own STL-
            # loading fallback. Latched (not re-tried every frame) so one
            # bad frame doesn't pay the exception cost for the whole video.
            _perspective_workspace_unavailable = True
    _draw_workspace_orthographic(ax, arm, q, target, trail, success_radius, d_mm, mesh_renderer)


# ── 3-DOF (shoulder_pan) workspace panel ─────────────────────────────────
#
# A SEPARATE, independent projection from the event-camera pipeline (per
# the approved plan) -- matplotlib's native Axes3D, a freely-posed 3/4-
# angle debug view, decoupled from wherever the simulated camera points.
# Deliberately a LINE/joint schematic (mirroring _draw_schematic_arm's own
# existing 2D fallback), not filled mesh polygons (Poly3DCollection) --
# this panel is purely a human-facing debug video, never fed into
# training (same "an accurate robot shape, not a CAD render" standard
# arm_mesh.py's own docstring already applies to the 2D case), and a line
# schematic sidesteps the real open question the plan flagged (whether
# the mesh's cached hull-candidate reduction still bounds the true shape
# under a genuinely 3D, non-planar pose) entirely, at the cost of losing
# the real link silhouettes.

def _draw_workspace_3d(ax, arm: SO101PlanarArm, q, target, trail, success_radius, d_mm):
    ax.clear()
    ax.set_title(f"workspace (3D)  d={d_mm:.1f}mm", fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)

    p_base = np.zeros(3)
    p0 = arm.pivot_xyz(q)
    p1 = arm.elbow_xyz(q)
    p2 = arm.fk(q)

    reach_center = np.array([arm.pan_x0, arm.pan_y0, p0[2]])
    margin = 0.05
    lo = reach_center - (arm.MAX_REACH + margin)
    hi = reach_center + (arm.MAX_REACH + margin)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass  # older matplotlib without set_box_aspect -- cosmetic only

    # Reach envelope: wireframe spheres at MIN/MAX_REACH around the pan
    # axis (the true reachable set is also bounded by pan's own angular
    # limits, +-1.919rad -- not a full sphere -- same "outer bound only"
    # simplification the 2D panel's own MIN/MAX_REACH circles already make
    # for q1/q2's limits, extended here rather than solved exactly).
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 12), np.linspace(0, np.pi, 8))
    for r, style in ((arm.MIN_REACH, dict(color="gray", linewidth=0.4, alpha=0.25)),
                     (arm.MAX_REACH, dict(color="gray", linewidth=0.4, alpha=0.25))):
        xs = reach_center[0] + r * np.cos(u) * np.sin(v)
        ys = reach_center[1] + r * np.sin(u) * np.sin(v)
        zs = reach_center[2] + r * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, **style)

    # Base -> pivot (fixed pedestal) -> elbow -> gripper, plus joints --
    # same construction as _draw_schematic_arm, in 3D.
    ax.plot(*zip(p_base, p0), color="darkslategray", linewidth=4)
    ax.plot(*zip(p0, p1), color="tab:orange", linewidth=6)
    ax.plot(*zip(p1, p2), color="tab:blue", linewidth=5)
    ax.scatter(*p0, color="black", s=25)
    ax.scatter(*p1, color="black", s=20)

    # Target + success radius (wireframe sphere, low-res) + trail.
    ax.scatter(*target, color="tab:red", marker="x", s=60, linewidth=2)
    su, sv = np.meshgrid(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 6))
    ax.plot_wireframe(target[0] + success_radius * np.cos(su) * np.sin(sv),
                       target[1] + success_radius * np.sin(su) * np.sin(sv),
                       target[2] + success_radius * np.cos(sv),
                       color="tab:green", linewidth=0.6, alpha=0.6)

    if len(trail) > 1:
        trail_arr = np.asarray(trail)
        n = len(trail_arr)
        for i in range(n - 1):
            alpha = 0.15 + 0.85 * (i / max(n - 2, 1))
            ax.plot(trail_arr[i:i + 2, 0], trail_arr[i:i + 2, 1], trail_arr[i:i + 2, 2],
                    color="tab:blue", alpha=alpha, linewidth=1.5)


def _draw_blank(ax, label):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=11, color="gray")


def _draw_image(ax, frame, title, cmap="gray", vmin=None, vmax=None):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    if frame is not None:
        ax.imshow(frame, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)


def _draw_spike_panel(ax, title, spk, parts):
    """One spike-raster "box" (critic+DA / D1+D2 / cortex). parts: list of
    (data_key, count_key, label, color), stacked bottom-to-top in the
    given order with a thin horizontal line between consecutive groups --
    the "boxes separated with a line" the user asked for applies to the
    row groups WITHIN a panel; the 3 panels themselves are already
    visually separated by being distinct subplots with their own axes
    border. Redrawn fresh from THIS step's own spk snapshot every frame
    (same "cheap per-frame snapshot" pattern _draw_image already uses for
    the pool-activity panel) -- no cross-frame state."""
    ax.clear()
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    if spk is None:
        return
    offset = 0
    for idx, (key, count_key, label, color) in enumerate(parts):
        times, rows = spk[key]
        n = spk[count_key]
        if len(times):
            ax.scatter(times, rows + offset, s=3, c=color, linewidths=0, label=label)
        offset += n
        if idx < len(parts) - 1 and offset > 0:
            ax.axhline(offset - 0.5, color="0.6", linewidth=0.8)
    ax.set_xlim(0, max(spk["window_ms"], 1.0))
    ax.set_ylim(-0.5, max(offset - 0.5, 0.5))
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc="upper right", markerscale=4, framealpha=0.6)


def _fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


def render_episode_video(out_path, arm, mode, steps, target, success, success_radius, fps=20):
    """Render one episode to `out_path` (mp4).

    steps: list of per-control-step dicts with keys:
        q (2,), d_mm, cum_reward, ep, step
        camera (H,W) uint8 or None    -- mode B only
        events (H,W,3) uint8 or None  -- mode B only, evsim_video.render_events output
        pool   (P,P) float or None    -- mode B only, pool spike counts
        camera_right/events_right/pool_right -- stereo_input only, right
            eye's own versions of the three panels above, permanently
            shown as a second row under camera/events/pool.
        spk -- dict or None, a sliding-window spike-raster snapshot (the
            last cfg.spike_panel_window_ms of biological time, not just
            this step's own slice -- see main.py's own comment there):
              window_ms: float, width of this frame's capture window
              cortex_exc/cortex_inh/critic/da/d1/d2: (times_ms, rows)
                  arrays (times relative to the window start, rows =
                  0-based neuron index local to that panel)
              n_cortex_exc/n_cortex_inh/n_critic/n_da/n_d1/n_d2: int,
                  row counts (d1/d2 already merged across every actor
                  channel into one combined panel)
            None keeps the original 1/4-panel layout -- see has_spk below.
    """
    # is_3d: a separate Axes3D projection for axes[0] only, decided once
    # at figure-construction time (a subplot's projection can't change
    # per-frame) -- everything else is untouched, and the non-stereo path
    # below still goes through the exact original plt.subplots(1,4,...)
    # call, unchanged.
    is_3d = getattr(arm, "n_joints", 2) == 3
    # is_stereo: a full second row (camera/events/pool, right eye) under
    # the existing row, only when at least one step actually carries a
    # camera_right frame -- every non-stereo config keeps the original
    # single-row 4-panel layout unchanged.
    is_stereo = any(s.get("camera_right") is not None for s in steps)
    # has_spk: 3 extra columns (critic+DA / D1+D2 / cortex spike rasters),
    # appended after whatever panels mode/is_stereo/is_3d already decided
    # on -- independent of all three, so it adds columns rather than
    # replacing any existing branch.
    has_spk = any(s.get("spk") is not None for s in steps)
    n_spk_cols = 3 if has_spk else 0

    # mode != "events": this task never produces camera/events/pool
    # frames at all (ground-truth-state tasks) -- a single, bigger
    # workspace panel instead of 3 permanently-blank "mode A" panels
    # every frame. is_stereo is always False here in practice
    # (camera_right is only ever populated under mode="events"), so this
    # check comes first and is exhaustive for the non-camera case.
    if mode != "events":
        if has_spk:
            # width_ratios: workspace gets 2.2x a spike panel's width
            # instead of an equal 1x share, and the figure is taller
            # (6in vs 5in) to give the now-larger
            # titles/legend (_TITLE_FONTSIZE/_LEGEND_FONTSIZE above) room
            # without crowding the data above/below it.
            fig = plt.figure(figsize=(7.0 + 2.0 * n_spk_cols, 6))
            gs = fig.add_gridspec(1, 1 + n_spk_cols,
                                   width_ratios=[2.2] + [1.0] * n_spk_cols)
            ax_workspace = fig.add_subplot(gs[0, 0], projection="3d" if is_3d else None)
            ax_critic_l = fig.add_subplot(gs[0, 1])
            ax_d1d2_l = fig.add_subplot(gs[0, 2])
            ax_cortex_l = fig.add_subplot(gs[0, 3])
        else:
            fig = plt.figure(figsize=(5.5, 5))
            ax_workspace = fig.add_subplot(1, 1, 1, projection="3d" if is_3d else None)
            ax_critic_l = ax_d1d2_l = ax_cortex_l = None
        ax_cam_l = ax_ev_l = ax_pool_l = None
        ax_cam_r = ax_ev_r = ax_pool_r = None
    elif is_stereo:
        # 2 rows x (4+n_spk_cols) cols -- workspace spans BOTH rows in
        # column 0 (a GridSpec merge, not an extra column); camera/
        # events/pool (left eye) in row 0, cols 1-3; their right-eye
        # counterparts directly underneath in row 1, cols 1-3. The 3
        # spike panels (if any), row 0 only, cols 4-6 -- spk data isn't
        # per-eye, so it never gets a row-1 counterpart.
        fig = plt.figure(figsize=(16 + 3.0 * n_spk_cols, 8))
        gs = fig.add_gridspec(2, 4 + n_spk_cols)
        ax_workspace = fig.add_subplot(gs[:, 0], projection="3d" if is_3d else None)
        ax_cam_l = fig.add_subplot(gs[0, 1])
        ax_ev_l = fig.add_subplot(gs[0, 2])
        ax_pool_l = fig.add_subplot(gs[0, 3])
        ax_cam_r = fig.add_subplot(gs[1, 1])
        ax_ev_r = fig.add_subplot(gs[1, 2])
        ax_pool_r = fig.add_subplot(gs[1, 3])
        if has_spk:
            ax_critic_l = fig.add_subplot(gs[0, 4])
            ax_d1d2_l = fig.add_subplot(gs[0, 5])
            ax_cortex_l = fig.add_subplot(gs[0, 6])
        else:
            ax_critic_l = ax_d1d2_l = ax_cortex_l = None
    else:
        # mode=="events", not stereo -- 1 row x (4+n_spk_cols) cols.
        # Previously a hardcoded plt.subplots(1,4,...)/is_3d special
        # case; unified onto one GridSpec (identical geometry/figsize
        # when n_spk_cols==0, so no behavior change for any existing
        # non-spk caller) since the only real difference between the
        # old is_3d/non-3d branches was the workspace axes' projection.
        fig = plt.figure(figsize=(16 + 3.0 * n_spk_cols, 4))
        gs = fig.add_gridspec(1, 4 + n_spk_cols)
        ax_workspace = fig.add_subplot(gs[0, 0], projection="3d" if is_3d else None)
        ax_cam_l = fig.add_subplot(gs[0, 1])
        ax_ev_l = fig.add_subplot(gs[0, 2])
        ax_pool_l = fig.add_subplot(gs[0, 3])
        ax_cam_r = ax_ev_r = ax_pool_r = None
        if has_spk:
            ax_critic_l = fig.add_subplot(gs[0, 4])
            ax_d1d2_l = fig.add_subplot(gs[0, 5])
            ax_cortex_l = fig.add_subplot(gs[0, 6])
        else:
            ax_critic_l = ax_d1d2_l = ax_cortex_l = None
    trail = []
    writer = None

    for i, s in enumerate(steps):
        trail.append(arm.fk(s["q"]))

        if is_3d:
            _draw_workspace_3d(ax_workspace, arm, s["q"], target, trail, success_radius, s["d_mm"])
        else:
            _draw_workspace(ax_workspace, arm, s["q"], target, trail, success_radius, s["d_mm"])

        if mode == "events":
            # vmin/vmax fixed: the camera panels are raw uint8
            # intensity frames on a fixed small set of values (scene.py's
            # BACKGROUND=128/EE_OBJECT=96/OBJECT=160/EE_TIP_MARKER=200) --
            # WITHOUT a fixed range, matplotlib auto-scales each frame's
            # gray levels to ITS OWN min/max, so a frame where the target
            # happens to be fully occluded (only some of the values
            # present -- a real, legitimate occlusion from that camera's
            # own angle, not a bug) gets re-stretched to fill black->white,
            # making the background flash white and everything look like
            # it's "moving" frame-to-frame even when the true positions
            # barely changed. A range of 0-255 avoided that flicker but
            # left the actual data (96-200) looking washed-out/low-contrast
            # for a human viewer, since it's confined to the ~40%-80% gray
            # band -- 110-175 stays FIXED (same anti-flicker property) and
            # deliberately CLIPS EE_OBJECT (96) to solid black and
            # EE_TIP_MARKER (200) to solid white (both already the most
            # distinct categories, don't need internal graduation) so the
            # full display range is spent maximizing contrast between
            # BACKGROUND (128) and OBJECT (160) specifically -- the two
            # closest values, and the one pairing (background vs. target)
            # a human viewer most needs to tell apart. "events" (RGB,
            # evsim_video.render_events) is unaffected either way (imshow
            # never colormap-normalizes a 3-channel image); "pool"
            # intentionally stays auto-scaled (a relative activity
            # heatmap, not a fixed-intensity image).
            _draw_image(ax_cam_l, s.get("camera"), "camera (left)" if is_stereo else "camera",
                        vmin=110, vmax=175)
            _draw_image(ax_ev_l, s.get("events"), "events")
            _draw_image(ax_pool_l, s.get("pool"), "pool activity", cmap="hot")
            if is_stereo:
                _draw_image(ax_cam_r, s.get("camera_right"), "camera (right)", vmin=110, vmax=175)
                _draw_image(ax_ev_r, s.get("events_right"), "events (right)")
                _draw_image(ax_pool_r, s.get("pool_right"), "pool activity (right)", cmap="hot")
        elif ax_cam_l is not None:
            # Only reachable when mode=="events" was expected but this
            # step carries no frames yet -- e.g. a transient/edge case.
            # The normal mode!="state" case has no panels to draw into at
            # all (ax_cam_l is None, see figure setup above), so there is
            # nothing to blank.
            _draw_blank(ax_cam_l, "mode A")
            _draw_blank(ax_ev_l, "mode A")
            _draw_blank(ax_pool_l, "mode A")
            if is_stereo:
                _draw_blank(ax_cam_r, "mode A")
                _draw_blank(ax_ev_r, "mode A")
                _draw_blank(ax_pool_r, "mode A")

        if ax_critic_l is not None:
            _draw_spike_panel(ax_critic_l, "critic + DA", s.get("spk"), [
                ("critic", "n_critic", "critic L1", "tab:blue"),
                ("da", "n_da", "DA", "tab:red"),
            ])
            _draw_spike_panel(ax_d1d2_l, "D1 + D2", s.get("spk"), [
                ("d1", "n_d1", "D1", "tab:green"),
                ("d2", "n_d2", "D2", "tab:orange"),
            ])
            _draw_spike_panel(ax_cortex_l, "cortex", s.get("spk"), [
                ("cortex_exc", "n_cortex_exc", "exc", "tab:purple"),
                ("cortex_inh", "n_cortex_inh", "inh", "tab:gray"),
            ])

        status = "SUCCESS" if success and i == len(steps) - 1 else ""
        fig.suptitle(
            f"ep {s['ep']}  step {s['step']}  d={s['d_mm']:.1f}mm  "
            f"cum_reward={s['cum_reward']:.2f}  {status}",
            fontsize=_SUPTITLE_FONTSIZE,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        frame = _fig_to_array(fig)
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
    plt.close(fig)


def _mean_std_band(ax, episodes, mean, std, label):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ax.plot(episodes, mean, label=label, linewidth=1.2)
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)


def _rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def render_summary(stats, output_folder, rolling_window=20):
    """Write summary.png at the end of a run: final distance per episode,
    success rate (rolling), cum reward, D1/D2/critic weight mean AND std
    (a flat mean proves nothing, log the variance), cortex active-fraction."""
    episodes = np.asarray(stats.get("episode", []))
    if len(episodes) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].plot(episodes, stats.get("final_d_mm", []))
    axes[0, 0].set_title("final distance (mm)")

    success = stats.get("success", np.zeros_like(episodes))
    axes[0, 1].plot(episodes, _rolling_mean(success, rolling_window))
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title(f"success rate (rolling {rolling_window})")

    axes[0, 2].plot(episodes, stats.get("cum_reward", []))
    axes[0, 2].axhline(0.0, color="gray", linewidth=0.7)
    axes[0, 2].set_title("cumulative reward")

    ax_d = axes[1, 0]
    for key in sorted(stats):
        if key.endswith("_w_mean") and (key.startswith("d1_") or key.startswith("d2_")):
            std_key = key.replace("_w_mean", "_w_std")
            _mean_std_band(ax_d, episodes, stats[key],
                           stats.get(std_key, np.zeros_like(episodes)), key[:-len("_w_mean")])
    ax_d.set_title("D1/D2 weights (mean +- std)")
    ax_d.legend(fontsize=7)

    ax_c = axes[1, 1]
    if "critic_w_mean" in stats:
        _mean_std_band(ax_c, episodes, stats["critic_w_mean"],
                       stats.get("critic_w_std", np.zeros_like(episodes)), "critic")
        ax_c.legend(fontsize=7)
    ax_c.set_title("critic weight (mean +- std)")

    axes[1, 2].plot(episodes, stats.get("cortex_active_fraction", []))
    axes[1, 2].axhline(0.5, color="red", linestyle="--", linewidth=0.7, label="saturation ceiling")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("cortex active fraction")
    axes[1, 2].legend(fontsize=7)

    for ax in axes.ravel():
        ax.set_xlabel("episode")

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "summary.png"), dpi=150)
    plt.close(fig)
