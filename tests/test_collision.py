"""Unit tests for the vectorised gripper-clearance collision check."""
from __future__ import annotations

import numpy as np

from parcel_transform_pipeline import Bin, check_grasp_clearance

TOP_DOWN = np.array([0.0, 0.0, -1.0])


def test_centre_grasp_is_clear():
    bin_ = Bin()
    res = check_grasp_clearance(np.array([[0.0, 0.0, 0.10]]), TOP_DOWN, bin_)
    assert bool(res["clear"][0]) is True


def test_grasp_outside_wall_is_blocked():
    bin_ = Bin(width=0.40)
    # x well beyond +x wall (0.20)
    res = check_grasp_clearance(np.array([[0.30, 0.0, 0.10]]), TOP_DOWN, bin_)
    assert bool(res["clear"][0]) is False
    assert bool(res["grasp_lateral_ok"][0]) is False


def test_grasp_below_floor_is_blocked():
    bin_ = Bin()
    res = check_grasp_clearance(np.array([[0.0, 0.0, 0.0]]), TOP_DOWN, bin_,
                                gripper_radius=0.03)
    assert bool(res["floor_ok"][0]) is False
    assert bool(res["clear"][0]) is False


def test_gripper_radius_shrinks_safe_region():
    bin_ = Bin(width=0.40)  # +x wall at 0.20
    pt = np.array([[0.185, 0.0, 0.10]])
    clear_small = check_grasp_clearance(pt, TOP_DOWN, bin_, gripper_radius=0.01)["clear"][0]
    clear_big = check_grasp_clearance(pt, TOP_DOWN, bin_, gripper_radius=0.05)["clear"][0]
    assert bool(clear_small) is True
    assert bool(clear_big) is False


def test_batch_shape_and_mix():
    bin_ = Bin()
    pts = np.array([
        [0.0, 0.0, 0.10],    # clear
        [0.30, 0.0, 0.10],   # blocked (wall)
        [0.05, -0.05, 0.12],  # clear
    ])
    res = check_grasp_clearance(pts, TOP_DOWN, bin_)
    assert res["clear"].shape == (3,)
    assert res["clear"].tolist() == [True, False, True]
