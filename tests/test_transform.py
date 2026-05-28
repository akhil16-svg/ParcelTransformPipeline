"""Unit tests for the Transform class: compose, inverse, apply."""
from __future__ import annotations

import numpy as np
import pytest

from parcel_transform_pipeline import Transform, make_homogeneous, rpy_to_rotation


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def random_transform(rng: np.random.Generator) -> Transform:
    rpy = rng.uniform(-np.pi, np.pi, size=3)
    t = rng.uniform(-2.0, 2.0, size=3)
    R = rpy_to_rotation(rpy)
    return Transform(make_homogeneous(R, t))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# ---------------------------------------------------------------------------
# rotation sanity
# ---------------------------------------------------------------------------
def test_rotation_is_orthonormal(rng):
    for _ in range(20):
        R = rpy_to_rotation(rng.uniform(-np.pi, np.pi, size=3))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_rotation_known_yaw():
    # 90 deg yaw about z maps +x -> +y
    R = rpy_to_rotation(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]),
                               np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_rpy_batch_matches_single(rng):
    rpy = rng.uniform(-np.pi, np.pi, size=(8, 3))
    batch = rpy_to_rotation(rpy)
    assert batch.shape == (8, 3, 3)
    for i in range(8):
        np.testing.assert_allclose(batch[i], rpy_to_rotation(rpy[i]), atol=1e-12)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def test_apply_identity_is_noop():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    np.testing.assert_allclose(Transform.identity().apply(pts), pts)


def test_apply_pure_translation():
    T = Transform.from_rpy_xyz(0, 0, 0, 1.0, -2.0, 3.0)
    np.testing.assert_allclose(T.apply(np.array([0.0, 0.0, 0.0])),
                               np.array([1.0, -2.0, 3.0]))


def test_apply_rotation_then_translation():
    # 90 deg yaw, then translate by (1,0,0)
    T = Transform.from_rpy_xyz(0, 0, np.pi / 2, 1.0, 0.0, 0.0)
    out = T.apply(np.array([1.0, 0.0, 0.0]))  # (1,0,0) -> (0,1,0) -> +(1,0,0)
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.0]), atol=1e-12)


def test_apply_single_vs_batch(rng):
    T = random_transform(rng)
    pts = rng.uniform(-3, 3, size=(15, 3))
    batch = T.apply(pts)
    assert batch.shape == (15, 3)
    for i in range(15):
        np.testing.assert_allclose(batch[i], T.apply(pts[i]), atol=1e-12)


def test_apply_single_returns_1d():
    out = Transform.identity().apply(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3,)


def test_apply_matches_manual_matmul(rng):
    T = random_transform(rng)
    pts = rng.uniform(-3, 3, size=(10, 3))
    homog = np.hstack([pts, np.ones((10, 1))])
    expected = (T.matrix @ homog.T).T[:, :3]
    np.testing.assert_allclose(T.apply(pts), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------
def test_compose_equals_matrix_product(rng):
    a, b = random_transform(rng), random_transform(rng)
    np.testing.assert_allclose(a.compose(b).matrix, a.matrix @ b.matrix, atol=1e-12)


def test_compose_operator_matches_method(rng):
    a, b = random_transform(rng), random_transform(rng)
    np.testing.assert_allclose((a @ b).matrix, a.compose(b).matrix, atol=1e-12)


def test_compose_is_associative(rng):
    a, b, c = (random_transform(rng) for _ in range(3))
    left = a.compose(b).compose(c)
    right = a.compose(b.compose(c))
    np.testing.assert_allclose(left.matrix, right.matrix, atol=1e-10)


def test_compose_applies_other_first(rng):
    a, b = random_transform(rng), random_transform(rng)
    pts = rng.uniform(-3, 3, size=(12, 3))
    chained = a.compose(b).apply(pts)
    stepwise = a.apply(b.apply(pts))
    np.testing.assert_allclose(chained, stepwise, atol=1e-10)


def test_compose_with_identity_is_noop(rng):
    a = random_transform(rng)
    I = Transform.identity()
    np.testing.assert_allclose(a.compose(I).matrix, a.matrix, atol=1e-12)
    np.testing.assert_allclose(I.compose(a).matrix, a.matrix, atol=1e-12)


# ---------------------------------------------------------------------------
# inverse
# ---------------------------------------------------------------------------
def test_inverse_yields_identity(rng):
    a = random_transform(rng)
    np.testing.assert_allclose(a.compose(a.inverse()).matrix, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(a.inverse().compose(a).matrix, np.eye(4), atol=1e-10)


def test_inverse_matches_numpy_inv(rng):
    a = random_transform(rng)
    np.testing.assert_allclose(a.inverse().matrix, np.linalg.inv(a.matrix), atol=1e-10)


def test_inverse_undoes_apply(rng):
    a = random_transform(rng)
    pts = rng.uniform(-5, 5, size=(20, 3))
    np.testing.assert_allclose(a.inverse().apply(a.apply(pts)), pts, atol=1e-10)


def test_inverse_of_inverse(rng):
    a = random_transform(rng)
    np.testing.assert_allclose(a.inverse().inverse().matrix, a.matrix, atol=1e-12)


# ---------------------------------------------------------------------------
# apply_poses (batch pose-stack transform used by the pipeline)
# ---------------------------------------------------------------------------
def test_apply_poses_batch(rng):
    a = random_transform(rng)
    poses = np.stack([random_transform(rng).matrix for _ in range(7)])
    out = a.apply_poses(poses)
    assert out.shape == (7, 4, 4)
    for i in range(7):
        np.testing.assert_allclose(out[i], a.matrix @ poses[i], atol=1e-12)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_bad_matrix_shape_raises():
    with pytest.raises(ValueError):
        Transform(np.eye(3))
