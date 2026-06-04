"""
Video / sequence inference: feed all images in a folder to the model in a
single forward pass (multi-view mode), then save per-frame results.

Unlike run_inference_folder.py which processes each image independently,
this script feeds the entire sequence at once so the model can leverage
cross-frame information.

Example:
    python src/testing/run_inference_video.py \
        --model_name mda_mog_sky_l2 \
        --img_path eval_results/teaser_imgs \
        --output_dir eval_results/video_inference/teaser
"""

import os
import sys
import torch
import argparse
import json
import re
import time
import numpy as np
from glob import glob
from PIL import Image
from PIL.ImageOps import exif_transpose

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.testing.utils.model_choice import choose_model, CONFIGS
from src.dust3r.utils.image import ImgNorm
from src.testing.eval_cut3r.video_depth.utils import save_depth_maps
from depth_anything_3.model.utils.transform import (
    pose_encoding_to_extri_intri,
    unproject_depth_map_to_point_map,
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _log(message):
    print(message, flush=True)


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _should_report(current, total, interval):
    return current == 1 or current == total or current % max(1, interval) == 0


class ProgressReporter:
    """Write machine-readable progress and mirror concise updates to stdout."""

    def __init__(self, path):
        self.path = path
        self.started_at = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def update(self, stage, message=None, current=None, total=None, status="running", **extra):
        elapsed_sec = round(time.time() - self.started_at, 2)
        payload = {
            "stage": stage,
            "status": status,
            "message": message or stage,
            "updated_at": _utc_now(),
            "elapsed_sec": elapsed_sec,
        }
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if current is not None and total:
            payload["percent"] = round(100.0 * float(current) / float(total), 2)
        payload.update(extra)

        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, self.path)

        progress = ""
        if current is not None and total:
            progress = f" {current}/{total} ({payload['percent']:.2f}%)"
        _log(f">> {stage}: {payload['message']}{progress}")


def _split_patterns(patterns):
    return [p.strip() for p in patterns.split(",") if p.strip()]


def _natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _sort_image_files(paths, mode):
    if mode == "lex":
        return sorted(paths)
    if mode == "mtime":
        return sorted(paths, key=lambda p: (os.path.getmtime(p), p))
    if mode == "natural":
        return sorted(paths, key=lambda p: (_natural_sort_key(p), p))
    raise ValueError(f"Unsupported sort mode: {mode}")


def discover_image_files(img_path, image_glob="*", image_regex=None, sort_mode="natural"):
    """Find image files while allowing project-specific naming conventions."""
    patterns = _split_patterns(image_glob)
    regex = re.compile(image_regex) if image_regex else None
    files = []

    for pattern in patterns:
        files.extend(glob(os.path.join(img_path, pattern), recursive=True))

    unique_files = []
    seen = set()
    for path in files:
        if path in seen or not os.path.isfile(path):
            continue
        if not path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            continue
        if regex and not regex.search(os.path.basename(path)):
            continue
        seen.add(path)
        unique_files.append(path)

    return _sort_image_files(unique_files, sort_mode)


def resolve_output_dir(base_output_dir, model_name, sharp_boundary):
    suffix = model_name + "_sharp_boundary" if sharp_boundary else model_name
    output_dir = os.path.join(base_output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_input_manifest(filelist, output_dir):
    manifest_path = os.path.join(output_dir, "input_files.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in filelist:
            f.write(f"{path}\n")
    return manifest_path


def resolve_progress_path(output_dir, progress_file):
    if not progress_file:
        return os.path.join(output_dir, "progress.json")
    if os.path.isabs(progress_file):
        return progress_file
    return os.path.join(output_dir, progress_file)


def _resize_pil_image_local(img, long_edge_size):
    src_long_edge = max(img.size)
    if src_long_edge > long_edge_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / src_long_edge)) for x in img.size)
    return img.resize(new_size, interp)


def load_images_for_eval_safe(
    folder_or_list, size, square_ok=False, verbose=True, crop=True,
    patch_size=16, img_norm=None, progress=None, progress_interval=25,
):
    if isinstance(folder_or_list, str):
        if verbose:
            _log(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        if verbose:
            _log(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    imgs = []
    norm = ImgNorm if img_norm is None else img_norm
    total_candidates = len(folder_content)

    for path in folder_content:
        if not path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            continue
        img = exif_transpose(Image.open(os.path.join(root, path))).convert("RGB")

        w_src, h_src = img.size
        if size == 224:
            img = _resize_pil_image_local(img, round(size * max(w_src / h_src, h_src / w_src)))
        else:
            img = _resize_pil_image_local(img, size)

        w_resized, h_resized = img.size
        cx, cy = w_resized // 2, h_resized // 2

        if size == 224:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                target_w = int(2 * half)
                target_h = int(2 * half)
                img = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            halfw = ((2 * cx) // patch_size) * (patch_size // 2)
            halfh = ((2 * cy) // patch_size) * (patch_size // 2)
            if (not square_ok) and (w_resized == h_resized):
                halfh = int(round(3 * halfw / 4))

            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:
                target_w = int(2 * halfw)
                target_h = int(2 * halfh)
                img = img.resize((target_w, target_h), Image.LANCZOS)

        w_out, h_out = img.size
        if verbose:
            _log(f" - adding {path} with resolution {w_src}x{h_src} --> {w_out}x{h_out}")

        imgs.append(
            dict(
                img=norm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
        if progress and _should_report(len(imgs), total_candidates, progress_interval):
            progress.update(
                "loading_images",
                "Loading and resizing images",
                current=len(imgs),
                total=total_candidates,
                source=str(path),
                output_size=[h_out, w_out],
            )

    assert imgs, "no images found at " + root
    if verbose:
        _log(f" (Found {len(imgs)} images)")
    return imgs


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run video/sequence inference with Depth-Anything-3."
    )
    parser.add_argument("--model_name", type=str, default="mda_mog_sky_l2",
                        help="name of the model (see src/testing/utils/model_choice.py)")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to folder of sequential images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save inference results")
    parser.add_argument("--size", type=int, default=512, help="Image size for inference")
    parser.add_argument("--image_glob", type=str, default="*",
                        help="Comma-separated glob patterns relative to --img_path")
    parser.add_argument("--image_regex", type=str, default=None,
                        help="Optional regex applied to each image basename")
    parser.add_argument("--sort_mode", type=str, default="natural",
                        choices=["natural", "lex", "mtime"],
                        help="How to order discovered frames")
    parser.add_argument("--scene_id", type=str, default=None,
                        help="Stable scene id for raw output filenames")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Progress JSON path. Relative paths are resolved under output_dir/model.")
    parser.add_argument("--progress_interval", type=int, default=25,
                        help="Frame/file interval for progress updates during load/save.")
    parser.add_argument("--verbose_images", type=int, default=0,
                        help="Print one line per loaded image when non-zero.")

    # Model configs
    parser.add_argument("--crop_center_112", type=int, default=0)
    parser.add_argument("--cam_inp", type=int, default=0)
    parser.add_argument("--gt_cam_output", type=int, default=0)
    parser.add_argument("--output_normalize", type=int, default=0)
    parser.add_argument("--output_double_layers", type=int, default=0)
    parser.add_argument("--output_sharp_boundary", type=int, default=0)
    parser.add_argument("--render_sideview", type=int, default=1,
                        help="Render sideview point cloud visualization")
    parser.add_argument("--max_chunk", type=int, default=64,
                        help="Max frames fed to the model per forward pass "
                             "(chunked when num_views > max_chunk). "
                             "Note: chunking breaks cross-chunk attention.")
    parser.add_argument("--gpu_cooldown_sec", type=float, default=0.0,
                        help="Sleep this many seconds after each chunk to share the GPU more gently.")
    parser.add_argument("--pcd_depth_min", type=float, default=1e-3,
                        help="Minimum depth (inclusive) for sideview point cloud.")
    parser.add_argument("--pcd_depth_max", type=float, default=float("inf"),
                        help="Maximum depth (exclusive) for sideview point cloud.")
    return parser


def prepare_views(
    img_list, size, patch_size, img_norm, progress=None,
    progress_interval=25, verbose_images=False,
):
    """Prepare views for ALL frames in the sequence (single inference call).

    The model attends across all views in a single forward pass and requires
    a uniform spatial size whose H and W are both multiples of ``patch_size``
    (14 or 16, returned by choose_model). To handle folders with mixed
    aspect ratios without distorting the geometry, every frame is first
    *center-cropped* to a common aspect ratio (locked by frame 0's loaded
    dims), then resized to a common (target_h, target_w) that's snapped down
    to a multiple of ``patch_size``.
    """
    images = load_images_for_eval_safe(
        img_list, size=size, crop=False, patch_size=patch_size, img_norm=img_norm,
        square_ok=True, progress=progress, progress_interval=progress_interval,
        verbose=verbose_images,
    )

    # Frame 0 fixes the aspect ratio and the final (target_h, target_w).
    # Snap down to multiples of patch_size required by the backbone.
    h0, w0 = images[0]["img"].shape[-2:]
    target_h = (h0 // patch_size) * patch_size
    target_w = (w0 // patch_size) * patch_size
    assert target_h > 0 and target_w > 0, (
        f"Target dims after patch-size snap are non-positive: "
        f"h0={h0}, w0={w0}, patch_size={patch_size}"
    )
    ar_target = target_h / target_w

    for img_data in images:
        img_t = img_data["img"]  # (1, 3, H, W)
        cur_h, cur_w = img_t.shape[-2:]

        # 1) Center-crop the largest region with aspect ratio == ar_target.
        if cur_h / cur_w > ar_target:
            crop_h = int(round(cur_w * ar_target))
            crop_w = cur_w
        else:
            crop_h = cur_h
            crop_w = int(round(cur_h / ar_target))
        top = (cur_h - crop_h) // 2
        left = (cur_w - crop_w) // 2
        img_t = img_t[..., top:top + crop_h, left:left + crop_w]

        # 2) Resize to (target_h, target_w) only if cropping didn't already
        #    land us there.
        if img_t.shape[-2:] != (target_h, target_w):
            img_t = torch.nn.functional.interpolate(
                img_t, size=(target_h, target_w),
                mode="bilinear", align_corners=False, antialias=True,
            )

        img_data["img"] = img_t
        img_data["true_shape"] = np.int32([[target_h, target_w]])

    views = []
    for i, img_data in enumerate(images):
        view = {
            "img": img_data["img"],
            "ray_map": torch.full(
                (img_data["img"].shape[0], 6,
                 img_data["img"].shape[-2], img_data["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(img_data["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4).astype(np.float32)
            ).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def _preds_to_cpu(obj):
    """Recursively move tensors in a predictions dict to CPU to free GPU memory."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _preds_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_preds_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_preds_to_cpu(v) for v in obj)
    return obj


def _concat_frame_dim(values):
    """Concatenate a list of per-chunk values along the frame axis.

    - Tensors with ndim >= 2: concat along dim=1 (batch=0, frame=1).
    - Tensors with ndim == 1: concat along dim=0 (assume frame dim is 0).
    - Lists: recurse element-wise when lengths match; otherwise flatten.
    - Other types (scalars, strings): keep first-chunk value.
    """
    if len(values) == 1:
        return values[0]
    head = values[0]
    if isinstance(head, torch.Tensor):
        if all(isinstance(v, torch.Tensor) for v in values):
            dim = 1 if head.ndim >= 2 else 0
            return torch.cat(values, dim=dim)
        return head
    if isinstance(head, list):
        if all(isinstance(v, list) and len(v) == len(head) for v in values):
            return [_concat_frame_dim([v[i] for v in values]) for i in range(len(head))]
        merged = []
        for v in values:
            merged.extend(v)
        return merged
    return head


def _merge_chunk_predictions(chunks):
    """Merge per-chunk predictions into a single predictions dict.

    Concatenates the fields this script reads downstream along the frame axis.
    `views` is kept as the first chunk's copy because its dict structure is
    complex; MoG visualization is invoked per-chunk separately.
    """
    if len(chunks) == 1:
        return chunks[0]

    merged = dict(chunks[0])
    for key in ("depth", "images", "pose_enc", "sky_mask"):
        if all(key in c for c in chunks):
            merged[key] = _concat_frame_dim([c[key] for c in chunks])

    if all("raw_preds" in c for c in chunks):
        raw_merged = dict(chunks[0]["raw_preds"])
        raw_keys = set()
        for c in chunks:
            raw_keys.update(c["raw_preds"].keys())
        for key in raw_keys:
            if not all(key in c["raw_preds"] for c in chunks):
                continue
            raw_merged[key] = _concat_frame_dim([c["raw_preds"][key] for c in chunks])
        merged["raw_preds"] = raw_merged

    return merged


def _images_to_uint8(images, is_ppd):
    imgs_hwc = images.permute(0, 2, 3, 1).cpu().numpy()
    if is_ppd:
        return (np.clip(imgs_hwc, 0, 1) * 255).astype(np.uint8)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (np.clip(imgs_hwc * std + mean, 0, 1) * 255).astype(np.uint8)


def save_raw_predictions(
    predictions, output_dir, scene_id, num_views, is_ppd, progress=None,
    progress_interval=25,
):
    raw_save_path = os.path.join(output_dir, "raw")
    os.makedirs(raw_save_path, exist_ok=True)

    depth_pred_all = predictions["depth"].squeeze(0)[:num_views].cpu().numpy()
    rgb_all = _images_to_uint8(predictions["images"].squeeze(0)[:num_views], is_ppd)

    for j in range(num_views):
        depth_pred_j = depth_pred_all[j]
        np.savez_compressed(
            os.path.join(raw_save_path, f"{scene_id.replace('/', '_')}_view_{j}.npz"),
            depth_pred=depth_pred_j,
            depth_gt=np.zeros_like(depth_pred_j),
            valid_mask=np.ones_like(depth_pred_j, dtype=bool),
            rgb=rgb_all[j],
        )
        if progress and _should_report(j + 1, num_views, progress_interval):
            progress.update(
                "saving_raw",
                "Saving raw per-view npz files",
                current=j + 1,
                total=num_views,
            )

    _log(f"Saved {num_views} raw per-view npz files to {raw_save_path}")
    return raw_save_path


def save_depth_outputs(
    predictions,
    filelist,
    output_dir,
    num_views,
    output_double_layers=False,
    output_sharp_boundary=False,
    progress=None,
    progress_interval=25,
):
    depth_maps = predictions["depth"].squeeze(0)
    if not output_double_layers:
        def _depth_progress(done, total, _path):
            if progress and _should_report(done, total, progress_interval):
                progress.update(
                    "saving_depth",
                    "Saving depth npy/png files",
                    current=done,
                    total=total,
                )

        save_depth_maps(
            None, output_dir, conf_self=None, depth_maps=depth_maps[:num_views].cpu(),
            progress_callback=_depth_progress,
        )
        return

    raw_layer_depths = predictions["raw_preds"]["depth"]

    if output_sharp_boundary:
        assert isinstance(raw_layer_depths, list)
        sharp_depth = predictions["depth"]
        mog_weight_raw = predictions["raw_preds"]["mog_weight_raw"]
        transparent_pixels = mog_weight_raw.sum(dim=-1) > 1.5
        opaque_pixels = mog_weight_raw.sum(dim=-1) <= 1.5
        sharp_depth = sharp_depth * opaque_pixels + transparent_pixels * raw_layer_depths[0]

        layer_depths = [sharp_depth.detach().squeeze(0)]
        for layer in [raw_layer_depths[-1]]:
            layer_tensor = layer.detach()
            if layer_tensor.ndim == 4 and layer_tensor.shape[0] == 1:
                layer_tensor = layer_tensor.squeeze(0)
            layer_depths.append(layer_tensor[:num_views])

    elif isinstance(raw_layer_depths, list):
        layer_depths = []
        for layer in [raw_layer_depths[0], raw_layer_depths[-1]]:
            layer_tensor = layer.detach()
            if layer_tensor.ndim == 4 and layer_tensor.shape[0] == 1:
                layer_tensor = layer_tensor.squeeze(0)
            elif layer_tensor.ndim != 3:
                continue
            layer_depths.append(layer_tensor[:num_views])
    else:
        raw_squeezed = raw_layer_depths.squeeze(0) if raw_layer_depths.ndim == 4 else raw_layer_depths
        layer_depths = [raw_squeezed] * 2

    for j, file_path in enumerate(filelist):
        name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        out_subdir = os.path.join(output_dir, name_no_ext)
        os.makedirs(out_subdir, exist_ok=True)

        fused_depth = depth_maps[j : j + 1]
        save_depth_maps(None, out_subdir, conf_self=None, depth_maps=fused_depth.cpu())
        if progress:
            progress.update(
                "saving_depth",
                "Saving layered depth outputs",
                current=j + 1,
                total=num_views,
            )

        for layer_idx, layer_tensor in enumerate(layer_depths):
            layer_out = os.path.join(out_subdir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_out, exist_ok=True)
            save_depth_maps(None, layer_out, conf_self=None, depth_maps=layer_tensor[j : j + 1].cpu())


def main():
    args = get_args_parser().parse_args()

    CONFIGS["crop_center_112"] = bool(args.crop_center_112)
    CONFIGS["cam_inp"] = bool(args.cam_inp)
    CONFIGS["gt_cam_output"] = bool(args.gt_cam_output)
    CONFIGS["output_normalize"] = bool(args.output_normalize)
    CONFIGS["output_double_layers"] = bool(args.output_double_layers)
    CONFIGS["output_sharp_boundary"] = bool(args.output_sharp_boundary)

    inference_size = args.size
    name_lower = args.model_name.lower()
    is_mog = "mog" in name_lower
    is_ppd = name_lower.startswith("ppd")

    output_dir = resolve_output_dir(args.output_dir, args.model_name, args.output_sharp_boundary)
    progress = ProgressReporter(resolve_progress_path(output_dir, args.progress_file))
    progress.update(
        "starting",
        "Starting sequence inference",
        model=args.model_name,
        output_dir=output_dir,
        size=inference_size,
        max_chunk=args.max_chunk,
        gpu_cooldown_sec=args.gpu_cooldown_sec,
    )

    try:
        _run(args, output_dir, progress, inference_size, is_mog, is_ppd)
    except Exception as exc:
        progress.update("failed", str(exc), status="failed")
        raise


def _run(args, output_dir, progress, inference_size, is_mog, is_ppd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress.update("loading_model", f"Loading model: {args.model_name}", device=str(device))
    loaded = choose_model(args.model_name)
    model = loaded.model
    patch_size = loaded.patch_size
    img_norm = loaded.img_norm
    model.to(device)
    model.eval()

    progress.update(
        "discovering_images",
        "Discovering input images",
        img_path=args.img_path,
        image_glob=args.image_glob,
        image_regex=args.image_regex,
        sort_mode=args.sort_mode,
    )
    filelist = discover_image_files(
        args.img_path,
        image_glob=args.image_glob,
        image_regex=args.image_regex,
        sort_mode=args.sort_mode,
    )

    if not filelist:
        raise FileNotFoundError(
            f"No images found in {args.img_path} "
            f"(image_glob={args.image_glob!r}, image_regex={args.image_regex!r})"
        )

    progress.update(
        "preparing_inputs",
        "Discovered images",
        current=0,
        total=len(filelist),
        first_image=filelist[0],
        last_image=filelist[-1],
    )
    manifest_path = save_input_manifest(filelist, output_dir)
    _log(f"Input manifest saved to {manifest_path}")

    # Prepare ALL views at once (the key difference from run_inference_folder.py)
    views = prepare_views(
        filelist, inference_size, patch_size, img_norm,
        progress=progress, progress_interval=args.progress_interval,
        verbose_images=bool(args.verbose_images),
    )

    num_views = len(filelist)
    max_chunk = max(1, args.max_chunk)
    total_chunks = (num_views + max_chunk - 1) // max_chunk

    # Chunked inference: the model sees at most max_chunk frames per forward pass.
    # Per-chunk predictions are moved to CPU to free GPU memory between chunks,
    # then concatenated along the frame dim.
    chunk_predictions = []
    for chunk_index, chunk_start in enumerate(range(0, num_views, max_chunk), start=1):
        chunk_end = min(chunk_start + max_chunk, num_views)
        progress.update(
            "inferencing",
            f"Inference on frames [{chunk_start}:{chunk_end}]",
            current=chunk_start,
            total=num_views,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        chunk_views = views[chunk_start:chunk_end]
        with torch.no_grad():
            chunk_pred = model.inference(
                chunk_views, device, is_mog=is_mog, use_sky_mask=True, **CONFIGS
            )
        chunk_predictions.append(_preds_to_cpu(chunk_pred))
        progress.update(
            "inferencing",
            f"Finished frames [{chunk_start}:{chunk_end}]",
            current=chunk_end,
            total=num_views,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        del chunk_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.gpu_cooldown_sec > 0 and chunk_end < num_views:
            time.sleep(args.gpu_cooldown_sec)

    progress.update("merging_predictions", "Merging chunk predictions")
    predictions = _merge_chunk_predictions(chunk_predictions)

    scene_id = args.scene_id or os.path.basename(os.path.normpath(args.img_path)) or "sequence"
    save_raw_predictions(
        predictions, output_dir, scene_id, num_views, is_ppd,
        progress=progress, progress_interval=args.progress_interval,
    )

    # Save per-frame depth maps
    save_depth_outputs(
        predictions,
        filelist,
        output_dir,
        num_views,
        output_double_layers=bool(args.output_double_layers),
        output_sharp_boundary=bool(args.output_sharp_boundary),
        progress=progress,
        progress_interval=args.progress_interval,
    )

    # MoG visualization: run per-chunk because views/raw_preds must stay
    # frame-consistent and we don't merge `views` above.
    # Sideview point cloud rendering
    if args.render_sideview:
        try:
            progress.update("saving_sideview", "Building sideview point cloud")
            with torch.no_grad():
                sv_extrinsic, sv_intrinsic = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], predictions["images"].shape[-2:]
                )
                sv_depth = predictions["depth"].cpu().numpy().squeeze(0)[:num_views]
                sv_ext_w2c_np = sv_extrinsic.cpu().numpy().squeeze(0)[:num_views]
                sv_K_np = sv_intrinsic.cpu().numpy().squeeze(0)[:num_views]

                sv_pts3d = unproject_depth_map_to_point_map(
                    sv_depth, sv_ext_w2c_np, sv_K_np,
                )

                sv_imgs = (
                    predictions["images"]
                    .squeeze(0)[:num_views]
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                if is_ppd:
                    sv_colors = np.clip(sv_imgs, 0, 1)
                else:
                    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    sv_colors = np.clip(sv_imgs * _std + _mean, 0, 1)

                # Build validity mask: depth range + sky filtering
                sv_masks = (
                    (sv_depth > args.pcd_depth_min) & (sv_depth < args.pcd_depth_max)
                ).astype(np.float32)

                # Extract sky mask from model predictions
                raw_preds = predictions["raw_preds"]
                n_real = predictions["depth"].shape[1]
                if "sky_mask" in predictions:
                    sky_mask = predictions["sky_mask"][0, :n_real].cpu().numpy() > 0.5
                elif "mog_weight_full" in raw_preds:
                    mwf = raw_preds["mog_weight_full"][0, :n_real]
                    sky_mask = (mwf.argmax(dim=-1) == mwf.shape[-1] - 1).cpu().numpy()
                elif "sky_mask" in raw_preds:
                    sky_mask = raw_preds["sky_mask"][0, :n_real].cpu().numpy() > 0.5
                else:
                    sky_mask = None

                if sky_mask is not None:
                    sv_masks[sky_mask[:num_views]] = 0.0
                    _log(f"Sky mask applied: {sky_mask[:num_views].sum()} sky pixels removed")

                # Save combined 3D point cloud (all frames merged)
                import open3d as o3d
                all_pts = []
                all_colors = []
                for fid in range(num_views):
                    mask = sv_masks[fid] > 0.5  # (H, W)
                    pts = sv_pts3d[fid][mask]  # (N, 3)
                    cols = sv_colors[fid][mask]  # (N, 3)
                    all_pts.append(pts)
                    all_colors.append(cols)
                all_pts = np.concatenate(all_pts, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                pcd.colors = o3d.utility.Vector3dVector(all_colors)
                pcd_path = os.path.join(output_dir, "pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                _log(f"Point cloud saved: {len(all_pts)} points -> {pcd_path}")

                # Persist per-frame extrinsics (world-to-camera, OpenCV) and
                # intrinsics alongside the point cloud so downstream renderers
                # can re-project back into camera space.
                n_sv = len(sv_depth)
                sv_ext_w2c_4x4 = np.zeros((n_sv, 4, 4), dtype=np.float64)
                sv_ext_w2c_4x4[:, :3, :] = sv_ext_w2c_np.astype(np.float64)
                sv_ext_w2c_4x4[:, 3, 3] = 1.0
                cameras_path = os.path.join(output_dir, "cameras.npz")
                np.savez_compressed(
                    cameras_path,
                    extrinsics_w2c=sv_ext_w2c_4x4,
                    intrinsics=sv_K_np.astype(np.float64),
                )
                _log(f"Cameras saved: {n_sv} views -> {cameras_path}")

        except Exception as e:
            _log(f"Warning: sideview rendering failed: {e}")

    progress.update("complete", f"Results saved to {output_dir}", current=num_views, total=num_views, status="complete")
    _log(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
