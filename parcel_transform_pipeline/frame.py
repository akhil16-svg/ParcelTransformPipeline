"""Named coordinate frames that share a common reference frame."""
from __future__ import annotations

import numpy as np

from .transform import Transform


class Frame:
    """A named coordinate frame whose pose is expressed in a reference frame.

    All frames in a chain (camera, robot base, destination bin) are stored
    with their pose relative to the same world/reference frame, which makes
    relative transforms between any two of them trivial to compute.
    """

    def __init__(
        self,
        name: str,
        transform: Transform | None = None,
        reference: str = "world",
    ) -> None:
        self.name = name
        self.transform = transform if transform is not None else Transform.identity()
        self.reference = reference

    def to_reference(self, points: np.ndarray) -> np.ndarray:
        """Map point(s) from this frame into the reference frame."""
        return self.transform.apply(points)

    def from_reference(self, points: np.ndarray) -> np.ndarray:
        """Map point(s) from the reference frame into this frame."""
        return self.transform.inverse().apply(points)

    def relative_to(self, other: "Frame") -> Transform:
        """Transform expressing *this* frame in ``other``'s frame.

        Maps points given in ``self`` into ``other``.  Assumes both frames are
        expressed in the same reference frame.
        """
        return other.transform.inverse().compose(self.transform)

    def __repr__(self) -> str:
        return f"Frame(name={self.name!r}, reference={self.reference!r}, {self.transform})"
