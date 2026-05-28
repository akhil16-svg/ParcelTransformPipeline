"""3D matplotlib visualisation of the frame chain, bin, and grasp poses."""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .collision import Bin  # noqa: E402
from .frame import Frame  # noqa: E402
from .transform import Transform  # noqa: E402

_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x=red, y=green, z=blue


def _draw_triad(ax, transform: Transform, scale: float, label: str | None = None) -> None:
    o = transform.translation
    R = transform.rotation
    for i in range(3):
        d = R[:, i] * scale
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], color=_AXIS_COLORS[i], linewidth=2)
    if label:
        ax.text(o[0], o[1], o[2] + scale * 0.25, label, fontsize=9, fontweight="bold")


def _draw_bin(ax, bin_: Bin, bin_pose: Transform) -> None:
    lower, upper = bin_.interior_bounds()
    xs, ys, zs = (lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])
    # corner index = i*4 + j*2 + k  (i->x, j->y, k->z)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    corners_w = bin_pose.apply(corners)
    for a, b in combinations(range(8), 2):
        if bin(a ^ b).count("1") == 1:  # differ in exactly one axis -> an edge
            p, q = corners_w[a], corners_w[b]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#666666", lw=1.1)


def visualize_pipeline(
    frames: Sequence[Frame],
    bin_: Bin,
    bin_pose: Transform,
    grasp_poses_world: np.ndarray,
    standoff_points_world: np.ndarray,
    clear_mask: np.ndarray,
    out_path: str = "pipeline_view.png",
) -> str:
    """Render frames, the bin, and grasp/approach poses; save to ``out_path``.

    Parameters
    ----------
    frames:
        Frames (camera, base, bin) to draw, each posed in the world frame.
    bin_, bin_pose:
        Bin geometry and its pose in the world frame.
    grasp_poses_world:
        ``(N, 4, 4)`` grasp poses in the world frame.
    standoff_points_world:
        ``(N, 3)`` pre-grasp points in the world frame.
    clear_mask:
        ``(N,)`` bool, ``True`` where the grasp clears the bin.
    """
    grasp_poses_world = np.asarray(grasp_poses_world, dtype=float).reshape(-1, 4, 4)
    standoff_points_world = np.asarray(standoff_points_world, dtype=float).reshape(-1, 3)
    clear_mask = np.asarray(clear_mask, dtype=bool).reshape(-1)
    grasp_pts = grasp_poses_world[:, :3, 3]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for fr in frames:
        _draw_triad(ax, fr.transform, scale=0.12, label=fr.name)

    _draw_bin(ax, bin_, bin_pose)

    # grasp orientation triads (small) + approach lines
    for i in range(grasp_poses_world.shape[0]):
        _draw_triad(ax, Transform(grasp_poses_world[i]), scale=0.05)
        s = standoff_points_world[i]
        g = grasp_pts[i]
        ax.plot([s[0], g[0]], [s[1], g[1]], [s[2], g[2]],
                color="#999999", lw=0.8, linestyle="--")

    ok = clear_mask
    if np.any(ok):
        ax.scatter(*grasp_pts[ok].T, c="#2ca02c", s=55, depthshade=True,
                   label="grasp OK", edgecolors="k", linewidths=0.4)
    if np.any(~ok):
        ax.scatter(*grasp_pts[~ok].T, c="#d62728", s=55, marker="X",
                   depthshade=True, label="grasp BLOCKED", edgecolors="k", linewidths=0.4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Parcel transform pipeline: camera \u2192 base \u2192 bin")
    ax.legend(loc="upper left")

    _set_equal_aspect(ax, grasp_pts, standoff_points_world,
                      np.array([fr.transform.translation for fr in frames]))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _set_equal_aspect(ax, *point_sets: np.ndarray) -> None:
    pts = np.vstack([p.reshape(-1, 3) for p in point_sets if p.size])
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    centre = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0 + 0.1
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    ax.set_box_aspect((1, 1, 1))
