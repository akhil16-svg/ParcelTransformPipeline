"""Parcel detections in the camera frame and a simple random simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .transform import make_homogeneous, rpy_to_rotation


@dataclass
class ParcelDetection:
    """A single parcel detection as reported by the camera.

    Attributes
    ----------
    position:
        ``(3,)`` xyz position in metres, expressed in the camera frame.
    orientation:
        ``(3,)`` roll/pitch/yaw in radians, expressed in the camera frame.
    confidence:
        Detector confidence in ``[0, 1]``.
    parcel_id:
        Integer identifier.
    """

    position: np.ndarray
    orientation: np.ndarray
    confidence: float = 1.0
    parcel_id: int = -1

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=float).reshape(3)

    def pose_matrix(self) -> np.ndarray:
        """4x4 homogeneous pose of this parcel in the camera frame."""
        R = rpy_to_rotation(self.orientation)
        return make_homogeneous(R, self.position)


def simulate_detections(
    n: int = 12,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = 7,
) -> List[ParcelDetection]:
    """Generate ``n`` pseudo-random parcel detections in the camera frame.

    Parcels sit on a conveyor roughly 0.9-1.1 m in front of a downward-facing
    camera.  ``x`` / ``y`` are spread wide enough that, once mapped into the
    destination bin, some parcels fall outside the safe interior -- giving a
    useful mix of passing and failing collision checks.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # camera looks down its +z axis; parcels are ~1 m down-range
    x = rng.uniform(-0.28, 0.28, size=(n, 1))
    y = rng.uniform(-0.22, 0.22, size=(n, 1))
    z = rng.uniform(0.90, 1.10, size=(n, 1))
    positions = np.hstack([x, y, z])

    # parcels lie mostly flat; yaw varies freely, roll/pitch are small jitter
    roll = rng.normal(0.0, 0.04, size=(n, 1))
    pitch = rng.normal(0.0, 0.04, size=(n, 1))
    yaw = rng.uniform(-np.pi, np.pi, size=(n, 1))
    orientations = np.hstack([roll, pitch, yaw])

    confidence = rng.uniform(0.65, 0.99, size=n)

    return [
        ParcelDetection(positions[i], orientations[i], float(confidence[i]), i)
        for i in range(n)
    ]


def stack_pose_matrices(detections: Iterable[ParcelDetection]) -> np.ndarray:
    """Stack detections into a single ``(N, 4, 4)`` array of pose matrices.

    The rotation matrices are built for the whole batch in one vectorised
    call; this array is what gets pushed through the transform chain.
    """
    detections = list(detections)
    positions = np.array([d.position for d in detections])  # (N, 3)
    orientations = np.array([d.orientation for d in detections])  # (N, 3)
    rotations = rpy_to_rotation(orientations)  # (N, 3, 3) -- vectorised
    return make_homogeneous(rotations, positions)  # (N, 4, 4) -- vectorised
