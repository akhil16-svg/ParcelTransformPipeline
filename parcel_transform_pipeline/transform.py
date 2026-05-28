"""Homogeneous 4x4 rigid-body transforms for the parcel pipeline.

Everything in this module is built on plain NumPy.  A :class:`Transform`
wraps a single 4x4 homogeneous matrix and supports ``compose``, ``inverse``
and ``apply``.  Batch helpers (``apply`` on ``(N, 3)`` points and
``apply_poses`` on ``(N, 4, 4)`` pose stacks) are fully vectorised so an
entire set of detections is processed in one NumPy call -- no Python loops.
"""
from __future__ import annotations

import numpy as np


def rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw angles to rotation matrices.

    Uses the ZYX (yaw-pitch-roll) intrinsic convention::

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Parameters
    ----------
    rpy:
        Shape ``(3,)`` for one orientation or ``(N, 3)`` for a batch.
        Columns are ``(roll, pitch, yaw)`` in radians.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` or ``(N, 3, 3)`` rotation matrices.
    """
    rpy = np.asarray(rpy, dtype=float)
    single = rpy.ndim == 1
    rpy = np.atleast_2d(rpy)

    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    n = rpy.shape[0]
    R = np.empty((n, 3, 3), dtype=float)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr

    return R[0] if single else R


def make_homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble 4x4 transform(s) from rotation(s) and translation(s).

    Accepts ``(3, 3)`` + ``(3,)`` (returns ``(4, 4)``) or ``(N, 3, 3)`` +
    ``(N, 3)`` (returns ``(N, 4, 4)``).
    """
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    single = rotation.ndim == 2

    R = rotation[None] if single else rotation
    t = translation[None] if single else translation

    n = R.shape[0]
    T = np.zeros((n, 4, 4), dtype=float)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T[0] if single else T


class Transform:
    """A rigid-body transform stored as a 4x4 homogeneous matrix.

    Semantics: a ``Transform`` maps points expressed in a *source* frame into
    a *target* frame.  Equivalently it is the pose of the source frame
    expressed in the target frame.
    """

    __slots__ = ("matrix",)

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        if matrix is None:
            matrix = np.eye(4)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {matrix.shape}")
        self.matrix = matrix

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.eye(4))

    @classmethod
    def from_rpy_xyz(
        cls, roll: float, pitch: float, yaw: float, x: float, y: float, z: float
    ) -> "Transform":
        R = rpy_to_rotation(np.array([roll, pitch, yaw]))
        return cls(make_homogeneous(R, np.array([x, y, z])))

    @classmethod
    def from_components(cls, rotation: np.ndarray, translation: np.ndarray) -> "Transform":
        return cls(make_homogeneous(rotation, translation))

    # ------------------------------------------------------------------
    # core operations
    # ------------------------------------------------------------------
    def compose(self, other: "Transform") -> "Transform":
        """Return ``self @ other`` (apply ``other`` first, then ``self``)."""
        return Transform(self.matrix @ other.matrix)

    def __matmul__(self, other: "Transform") -> "Transform":
        return self.compose(other)

    def inverse(self) -> "Transform":
        """Closed-form inverse of a rigid transform (no general solve)."""
        R = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = R.T
        inv[:3, 3] = -R.T @ t
        return Transform(inv)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform 3D point(s).

        ``points`` may be ``(3,)`` or ``(N, 3)``.  The batch case is a single
        matrix multiply -- there is no Python-level loop over points.
        """
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)  # (N, 3)
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 4)
        out = homog @ self.matrix.T  # (N, 4)  == (M @ p)^T per row
        out = out[:, :3]
        return out[0] if single else out

    def apply_poses(self, poses: np.ndarray) -> np.ndarray:
        """Left-multiply a batch of pose matrices: ``self.matrix @ poses``.

        ``poses`` may be ``(4, 4)`` or ``(N, 4, 4)``.  Uses broadcasting
        matmul so the whole stack is transformed at once.
        """
        poses = np.asarray(poses, dtype=float)
        single = poses.ndim == 2
        poses = poses.reshape(-1, 4, 4)
        out = self.matrix @ poses  # (4,4) @ (N,4,4) -> (N,4,4) via broadcasting
        return out[0] if single else out

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3].copy()

    def __repr__(self) -> str:
        t = self.translation
        return f"Transform(t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}])"
