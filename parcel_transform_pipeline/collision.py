"""Destination-bin geometry and a vectorised gripper clearance check."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Bin:
    """Axis-aligned destination bin with an open top.

    Coordinates are expressed in the *bin frame*: origin at the centre of the
    floor, ``z`` pointing up.  The bin has solid walls of height ``height``
    and an open top through which the gripper descends.
    """

    width: float = 0.40   # interior extent along x (m)
    depth: float = 0.30   # interior extent along y (m)
    height: float = 0.30  # wall height along z (m)

    def interior_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper ``(3,)`` corners of the interior volume."""
        hx, hy = self.width / 2.0, self.depth / 2.0
        lower = np.array([-hx, -hy, 0.0])
        upper = np.array([hx, hy, self.height])
        return lower, upper


def check_grasp_clearance(
    grasp_points_bin: np.ndarray,
    approach_vector: np.ndarray,
    bin_: Bin,
    gripper_radius: float = 0.03,
    standoff: float = 0.10,
) -> Dict[str, np.ndarray]:
    """Check whether each grasp clears the bin walls and floor.

    The gripper starts at a pre-grasp point ``standoff`` metres back along
    ``-approach_vector`` and travels in a straight line to the grasp point.
    A grasp is collision-free when:

    * the grasp point clears the walls laterally by ``gripper_radius``,
    * the point where the approach path crosses the bin rim (``z = height``)
      also clears the walls laterally (matters for angled approaches),
    * the grasp point sits above the floor by at least ``gripper_radius``,
    * the grasp point is at or below the rim (it is actually inside the bin).

    Because the bin interior is convex, checking the rim-entry point and the
    grasp point is sufficient to guarantee the whole in-wall segment is clear.

    Everything is computed across all ``N`` grasps at once with NumPy.

    Parameters
    ----------
    grasp_points_bin:
        ``(N, 3)`` grasp positions expressed in the bin frame.
    approach_vector:
        ``(3,)`` unit direction the gripper travels while approaching
        (e.g. ``(0, 0, -1)`` for a top-down descent).
    bin_:
        Bin geometry.
    gripper_radius:
        Half-width of the gripper, used to shrink the safe region.
    standoff:
        Distance of the pre-grasp point from the grasp point.

    Returns
    -------
    dict
        ``clear`` ``(N,)`` bool mask plus the intermediate masks and points
        used to derive it.
    """
    grasp = np.atleast_2d(np.asarray(grasp_points_bin, dtype=float))  # (N, 3)
    a = np.asarray(approach_vector, dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError("approach_vector must be non-zero")
    a = a / norm

    standoff_points = grasp - a * standoff  # (N, 3) start of approach
    seg = grasp - standoff_points           # (N, 3) == a * standoff

    lower, upper = bin_.interior_bounds()
    r = gripper_radius
    lo_xy = lower[:2] + r
    hi_xy = upper[:2] - r
    height = upper[2]

    # Rim-entry point: where the descent crosses z == height.
    dz = seg[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_rim = (height - standoff_points[:, 2]) / dz
    s_rim = np.clip(np.nan_to_num(s_rim, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    rim_points = standoff_points + s_rim[:, None] * seg  # (N, 3)

    def lateral_ok(points: np.ndarray) -> np.ndarray:
        return np.all((points[:, :2] >= lo_xy) & (points[:, :2] <= hi_xy), axis=1)

    grasp_lateral_ok = lateral_ok(grasp)
    rim_lateral_ok = lateral_ok(rim_points)
    floor_ok = grasp[:, 2] >= r
    below_rim = grasp[:, 2] <= height

    clear = grasp_lateral_ok & rim_lateral_ok & floor_ok & below_rim

    return {
        "clear": clear,
        "grasp_lateral_ok": grasp_lateral_ok,
        "rim_lateral_ok": rim_lateral_ok,
        "floor_ok": floor_ok,
        "below_rim": below_rim,
        "standoff_points": standoff_points,
        "rim_points": rim_points,
    }
