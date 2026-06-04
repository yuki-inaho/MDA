"""Export saved MDA sequence inference results to a Rerun recording.

The default demo result does not contain global camera poses unless side-view
export was enabled during inference.  This exporter therefore logs a loopable
per-frame camera-space point cloud, RGB image, and depth image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

try:
    from result_viewer import ResultScene, _build_point_cloud_frame
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from src.testing.result_viewer import ResultScene, _build_point_cloud_frame


DEFAULT_APP_ID = "mda_result"


def _frame_indices(
    frame_count: int,
    start_frame: int,
    end_frame: int,
    frame_stride: int,
    max_frames: int,
) -> list[int]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    start = max(0, int(start_frame))
    end = frame_count if end_frame < 0 else min(frame_count, int(end_frame))
    if start >= end:
        raise ValueError(f"empty frame range: start={start}, end={end}, frame_count={frame_count}")
    indices = list(range(start, end, frame_stride))
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def _make_blueprint() -> object | None:
    try:
        import rerun.blueprint as rrb
    except Exception:
        return None

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="camera"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera/image"),
                rrb.Spatial2DView(origin="camera/depth"),
            ),
        ),
        collapse_panels=True,
    )


def _iter_with_progress(indices: Iterable[int], enabled: bool) -> Iterable[int]:
    if not enabled:
        return indices
    return tqdm(list(indices), desc="writing Rerun frames", unit="frame")


def _log_frame(rr: object, scene: ResultScene, index: int, args: argparse.Namespace) -> None:
    raw = scene.raw(index)
    depth = raw["depth_pred"].astype(np.float32, copy=False)
    rgb = raw["rgb"].astype(np.uint8, copy=False)
    point_cloud = _build_point_cloud_frame(
        scene=scene,
        index=index,
        max_points=args.max_points,
        stride=args.stride,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_percentile=args.depth_percentile,
        focal_scale=args.focal_scale,
        filter_depth_edges=args.filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        mask_black=args.mask_black,
        mask_white=args.mask_white,
    )

    height, width = depth.shape
    fx = fy = max(height, width) * max(float(args.focal_scale), 0.01)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    rr.set_time("frame", sequence=index)
    rr.log(
        "camera",
        rr.Pinhole(
            image_from_camera=intrinsics,
            width=width,
            height=height,
            image_plane_distance=0.02,
        ),
    )
    rr.log("camera/image", rr.Image(rgb))
    rr.log("camera/depth", rr.DepthImage(depth))
    rr.log(
        "camera/points",
        rr.Points3D(point_cloud.points, colors=point_cloud.colors, radii=args.point_radius),
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an MDA result directory to a Rerun .rrd file.")
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Directory containing frame_*.png and raw/*.npz, usually eval_results/.../<model_name>",
    )
    parser.add_argument(
        "--rrd",
        type=Path,
        default=None,
        help="Output .rrd path. Defaults to <result_dir>/mda_result.rrd.",
    )
    parser.add_argument("--app_id", default=DEFAULT_APP_ID)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1, help="Exclusive end frame. -1 means all frames.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max_points", type=int, default=15000)
    parser.add_argument("--stride", type=int, default=0, help="0 chooses a stride from --max_points.")
    parser.add_argument("--depth_min", type=float, default=0.0)
    parser.add_argument("--depth_max", type=float, default=float("inf"))
    parser.add_argument(
        "--depth_percentile",
        type=float,
        default=95.0,
        help="Far-depth percentile clip for point clouds. Use 0 or 100 to disable.",
    )
    parser.add_argument("--focal_scale", type=float, default=1.2)
    parser.add_argument("--filter_depth_edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--mask_black", action="store_true")
    parser.add_argument("--mask_white", action="store_true")
    parser.add_argument("--point_radius", type=float, default=0.0008)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "rerun-sdk is optional. Run this command with `uv run --extra viz ...` "
            "or install it with `uv sync --extra viz`."
        ) from exc

    scene = ResultScene.from_dir(args.result_dir)
    indices = _frame_indices(
        frame_count=scene.frame_count,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    rrd_path = (args.rrd or (scene.root / "mda_result.rrd")).expanduser().resolve()
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(args.app_id, spawn=False)
    blueprint = _make_blueprint()
    if blueprint is not None:
        rr.save(str(rrd_path), default_blueprint=blueprint)
    else:
        rr.save(str(rrd_path))

    for index in _iter_with_progress(indices, args.progress):
        _log_frame(rr, scene, index, args)

    rr.disconnect()
    print(
        f"wrote {rrd_path} "
        f"frames={len(indices)} max_points={args.max_points} depth_percentile={args.depth_percentile}"
    )


if __name__ == "__main__":
    main()
