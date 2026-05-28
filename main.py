"""End-to-end parcel transform pipeline.

Run with::

    python main.py

Pipeline:
    1. Define the frame chain  camera -> robot base (world) -> destination bin.
    2. Simulate random parcel detections in the camera frame.
    3. Push every detection pose through the chain in single vectorised
       NumPy operations (no Python loops over detections).
    4. Run a vectorised gripper-clearance collision check in the bin frame.
    5. Print a summary and save a 3D visualisation.
"""
from __future__ import annotations

import argparse

import numpy as np

from parcel_transform_pipeline import (
    Bin,
    Frame,
    Transform,
    check_grasp_clearance,
    simulate_detections,
    stack_pose_matrices,
)
from parcel_transform_pipeline.visualize import visualize_pipeline


def build_frames() -> tuple[Frame, Frame, Frame]:
    """Construct the camera, robot-base, and bin frames in the world frame.

    World frame == robot base.  The destination bin sits on the table, offset
    in +x from the base.  The camera is mounted ~1.2 m above the bin looking
    straight down, so parcel detections project into the bin volume.
    """
    base = Frame("robot_base", Transform.identity(), reference="world")

    # Bin offset in +x, slightly rotated about z.
    bin_x = 0.50
    bin_frame = Frame(
        "dest_bin",
        Transform.from_rpy_xyz(roll=0.0, pitch=0.0, yaw=0.12, x=bin_x, y=0.0, z=0.0),
        reference="world",
    )

    # Camera directly above the bin, looking down: rotate 180 deg about
    # world-x so the camera's +z axis points toward -z world (down at the bin).
    camera = Frame(
        "camera",
        Transform.from_rpy_xyz(roll=np.pi, pitch=0.0, yaw=0.0, x=bin_x, y=0.0, z=1.20),
        reference="world",
    )
    return camera, base, bin_frame


def run(n: int = 12, seed: int | None = 7, out_path: str = "pipeline_view.png") -> dict:
    camera, base, bin_frame = build_frames()
    bin_geom = Bin(width=0.40, depth=0.30, height=0.30)

    # --- 1. detections in the camera frame ----------------------------
    detections = simulate_detections(n=n, seed=seed)
    det_poses_cam = stack_pose_matrices(detections)  # (N, 4, 4)

    # --- 2. transform chain (single vectorised ops) -------------------
    # camera -> world (robot base)
    T_world_cam = camera.transform
    world_poses = T_world_cam.apply_poses(det_poses_cam)  # one matmul, (N,4,4)

    # world -> bin frame
    T_bin_world = bin_frame.transform.inverse()
    bin_poses = T_bin_world.apply_poses(world_poses)  # one matmul, (N,4,4)

    grasp_pts_bin = bin_poses[:, :3, 3]      # (N, 3) grasp positions in bin frame
    grasp_pts_world = world_poses[:, :3, 3]  # (N, 3) grasp positions in world frame

    # --- 3. collision check (top-down approach in the bin frame) ------
    approach_bin = np.array([0.0, 0.0, -1.0])
    standoff = 0.10
    result = check_grasp_clearance(
        grasp_pts_bin, approach_bin, bin_geom, gripper_radius=0.03, standoff=standoff
    )
    clear = result["clear"]

    # standoff points back to world frame for plotting
    standoff_world = bin_frame.transform.apply(result["standoff_points"])

    # --- 4. report ----------------------------------------------------
    print(f"\nSimulated {n} parcel detection(s); chained camera -> base -> bin.\n")
    header = f"{'id':>3}  {'conf':>5}  {'bin x':>7} {'bin y':>7} {'bin z':>7}  {'verdict':>8}"
    print(header)
    print("-" * len(header))
    for d, p, ok in zip(detections, grasp_pts_bin, clear):
        verdict = "CLEAR" if ok else "BLOCKED"
        print(f"{d.parcel_id:>3}  {d.confidence:>5.2f}  "
              f"{p[0]:>7.3f} {p[1]:>7.3f} {p[2]:>7.3f}  {verdict:>8}")
    n_clear = int(clear.sum())
    print(f"\n{n_clear}/{n} grasps clear the bin walls "
          f"({100.0 * n_clear / n:.0f}%).")

    # --- 5. visualise -------------------------------------------------
    saved = visualize_pipeline(
        frames=[camera, base, bin_frame],
        bin_=bin_geom,
        bin_pose=bin_frame.transform,
        grasp_poses_world=world_poses,
        standoff_points_world=standoff_world,
        clear_mask=clear,
        out_path=out_path,
    )
    print(f"3D visualisation saved to: {saved}\n")

    return {
        "detections": detections,
        "world_poses": world_poses,
        "bin_poses": bin_poses,
        "clear": clear,
        "image": saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parcel transform pipeline.")
    parser.add_argument("-n", "--num", type=int, default=12, help="number of parcels")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (use -1 for random)")
    parser.add_argument("--out", type=str, default="pipeline_view.png", help="output image")
    args = parser.parse_args()
    seed = None if args.seed == -1 else args.seed
    run(n=args.num, seed=seed, out_path=args.out)


if __name__ == "__main__":
    main()
