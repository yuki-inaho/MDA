"""Browser viewer for saved MDA sequence inference results.

This is intentionally a results viewer, not another inference path.  It reads
the files produced by ``run_inference_video.py``:

* ``frame_XXXX.png``: colored depth image
* ``frame_XXXX.npy``: raw depth map
* ``raw/*_view_N.npz``: per-frame depth/rgb payload

The point cloud view uses a per-frame camera-space unprojection because the
default demo output does not include global camera poses unless side-view export
was enabled.  It is still useful for quick inspection of geometry, depth edges,
and temporal stability.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image


SUPPORTED_DEPTH_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _view_index(path: Path) -> int:
    match = re.search(r"_view_(\d+)\.npz$", path.name)
    if match:
        return int(match.group(1))
    return int(re.search(r"(\d+)", path.stem).group(1))


def _frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return int(re.search(r"(\d+)", path.stem).group(1))


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite_float(value: float | None, default: float) -> float:
    if value is None:
        return default
    if not math.isfinite(float(value)):
        return default
    return float(value)


@lru_cache(maxsize=24)
def _load_raw_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


@dataclass(frozen=True)
class ResultScene:
    root: Path
    raw_paths: tuple[Path, ...]
    depth_png_paths: tuple[Path, ...]
    depth_npy_paths: tuple[Path, ...]
    input_files: tuple[str, ...]

    @classmethod
    def from_dir(cls, result_dir: str | os.PathLike[str]) -> "ResultScene":
        root = Path(result_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"result_dir not found: {root}")

        raw_dir = root / "raw"
        raw_paths = tuple(sorted(raw_dir.glob("*.npz"), key=_view_index)) if raw_dir.exists() else ()
        depth_png_paths = tuple(
            sorted(
                [p for p in root.glob("frame_*") if p.suffix.lower() in SUPPORTED_DEPTH_IMAGE_EXTENSIONS],
                key=_frame_index,
            )
        )
        depth_npy_paths = tuple(sorted(root.glob("frame_*.npy"), key=_frame_index))
        input_files = tuple(_read_manifest(root / "input_files.txt"))

        if not raw_paths:
            raise FileNotFoundError(f"No raw/*.npz files found under {root}")
        if not depth_png_paths:
            raise FileNotFoundError(f"No frame_*.png depth images found under {root}")
        if len(raw_paths) != len(depth_png_paths):
            raise ValueError(
                f"raw/depth count mismatch: raw={len(raw_paths)}, depth_png={len(depth_png_paths)}"
            )

        return cls(
            root=root,
            raw_paths=raw_paths,
            depth_png_paths=depth_png_paths,
            depth_npy_paths=depth_npy_paths,
            input_files=input_files,
        )

    @property
    def frame_count(self) -> int:
        return len(self.raw_paths)

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= self.frame_count:
            raise HTTPException(status_code=404, detail=f"frame index out of range: {index}")

    def raw(self, index: int) -> dict[str, np.ndarray]:
        self._check_index(index)
        return _load_raw_npz(str(self.raw_paths[index]))

    def input_name(self, index: int) -> str:
        if index < len(self.input_files):
            return self.input_files[index]
        return self.raw_paths[index].name

    def metadata(self) -> dict[str, Any]:
        first = self.raw(0)
        depth = first["depth_pred"]
        rgb = first["rgb"]
        progress_path = self.root / "progress.json"
        progress = None
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                progress = None
        return {
            "result_dir": str(self.root),
            "frame_count": self.frame_count,
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "has_progress": progress is not None,
            "progress": progress,
            "raw_count": len(self.raw_paths),
            "depth_png_count": len(self.depth_png_paths),
            "depth_npy_count": len(self.depth_npy_paths),
            "first_input": self.input_name(0),
            "last_input": self.input_name(self.frame_count - 1),
            "rgb_shape": list(rgb.shape),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
        }


def _depth_edge_mask(depth: np.ndarray, rtol: float = 0.03, kernel: int = 3) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(depth, ((pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel):
        for x in range(kernel):
            window = padded[y : y + depth.shape[0], x : x + depth.shape[1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return relative_jump > rtol


def _auto_stride(height: int, width: int, max_points: int) -> int:
    max_points = max(1, int(max_points))
    return max(1, int(math.ceil(math.sqrt((height * width) / max_points))))


def _to_png_response(image: np.ndarray) -> Response:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _point_cloud_payload(
    scene: ResultScene,
    index: int,
    max_points: int,
    stride: int,
    depth_min: float,
    depth_max: float,
    focal_scale: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    mask_black: bool,
    mask_white: bool,
) -> dict[str, Any]:
    raw = scene.raw(index)
    depth = raw["depth_pred"].astype(np.float32, copy=False)
    rgb = raw["rgb"].astype(np.uint8, copy=False)
    valid = raw.get("valid_mask")
    if valid is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = valid.astype(bool, copy=False)

    height, width = depth.shape
    stride = int(stride) if stride and stride > 0 else _auto_stride(height, width, max_points)
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    colors = rgb[::stride, ::stride]
    keep = valid[::stride, ::stride] & np.isfinite(z)

    depth_min = _finite_float(depth_min, float(np.nanmin(depth)))
    depth_max = _finite_float(depth_max, float(np.nanmax(depth)))
    keep &= (z >= depth_min) & (z <= depth_max)
    if filter_depth_edges:
        keep &= ~_depth_edge_mask(depth, rtol=depth_edge_rtol)[::stride, ::stride]
    if mask_black:
        keep &= colors.sum(axis=-1) >= 16
    if mask_white:
        keep &= ~((colors[..., 0] > 240) & (colors[..., 1] > 240) & (colors[..., 2] > 240))

    fx = fy = max(height, width) * max(float(focal_scale), 0.01)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    x = (xs.astype(np.float32) - cx) / fx * z
    y = -(ys.astype(np.float32) - cy) / fy * z
    points = np.stack([x, y, -z], axis=-1)[keep]
    colors = colors[keep]

    if len(points) > max_points:
        sample = np.linspace(0, len(points) - 1, int(max_points)).astype(np.int64)
        points = points[sample]
        colors = colors[sample]

    if len(points):
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center = (bbox_min + bbox_max) * 0.5
        extent = float(np.max(bbox_max - bbox_min))
        display_scale = 2.0 / max(extent, 1e-6)
        depth_stats = {
            "min": float(np.nanmin(z[keep])),
            "max": float(np.nanmax(z[keep])),
            "mean": float(np.nanmean(z[keep])),
        }
    else:
        bbox_min = bbox_max = center = np.zeros(3, dtype=np.float32)
        display_scale = 1.0
        depth_stats = {"min": None, "max": None, "mean": None}

    points = np.ascontiguousarray(points.astype(np.float32))
    colors = np.ascontiguousarray(colors.astype(np.uint8))
    return {
        "frame": index,
        "input": scene.input_name(index),
        "count": int(len(points)),
        "stride": int(stride),
        "height": int(height),
        "width": int(width),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_stats": depth_stats,
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "display_scale": float(display_scale),
        "points_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(colors.tobytes()).decode("ascii"),
    }


def create_app(scene: ResultScene) -> FastAPI:
    app = FastAPI(title="MDA Result Viewer")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return VIEWER_HTML

    @app.get("/api/scene")
    def api_scene() -> dict[str, Any]:
        return scene.metadata()

    @app.get("/api/frame/{index}/rgb.png")
    def api_rgb(index: int) -> Response:
        raw = scene.raw(index)
        return _to_png_response(raw["rgb"].astype(np.uint8, copy=False))

    @app.get("/api/frame/{index}/depth.png")
    def api_depth(index: int) -> FileResponse:
        scene._check_index(index)
        return FileResponse(scene.depth_png_paths[index], media_type="image/png")

    @app.get("/api/frame/{index}/stats")
    def api_frame_stats(index: int) -> dict[str, Any]:
        raw = scene.raw(index)
        depth = raw["depth_pred"].astype(np.float32, copy=False)
        valid = raw.get("valid_mask", np.ones(depth.shape, dtype=bool)).astype(bool)
        finite = valid & np.isfinite(depth)
        values = depth[finite]
        return {
            "frame": index,
            "input": scene.input_name(index),
            "shape": list(depth.shape),
            "valid_pixels": int(finite.sum()),
            "depth_min": float(values.min()) if values.size else None,
            "depth_max": float(values.max()) if values.size else None,
            "depth_mean": float(values.mean()) if values.size else None,
            "depth_p01": float(np.percentile(values, 1)) if values.size else None,
            "depth_p99": float(np.percentile(values, 99)) if values.size else None,
        }

    @app.get("/api/frame/{index}/pointcloud")
    def api_pointcloud(
        index: int,
        max_points: int = Query(40000, ge=1000, le=200000),
        stride: int = Query(0, ge=0, le=64),
        depth_min: float = Query(0.0),
        depth_max: float = Query(float("inf")),
        focal_scale: float = Query(1.2, gt=0.01, le=10.0),
        filter_depth_edges: bool = Query(True),
        depth_edge_rtol: float = Query(0.03, gt=0.0, le=1.0),
        mask_black: bool = Query(False),
        mask_white: bool = Query(False),
    ) -> dict[str, Any]:
        scene._check_index(index)
        return _point_cloud_payload(
            scene=scene,
            index=index,
            max_points=max_points,
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            focal_scale=focal_scale,
            filter_depth_edges=filter_depth_edges,
            depth_edge_rtol=depth_edge_rtol,
            mask_black=mask_black,
            mask_white=mask_white,
        )

    return app


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MDA Result Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111416;
      --panel: #1b2024;
      --panel-2: #23292f;
      --line: #343c43;
      --text: #eef2f5;
      --muted: #aeb8c2;
      --accent: #5cc8a7;
      --warn: #e6b15b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 52px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: #15191d;
    }
    h1 {
      font-size: 15px;
      margin: 0;
      font-weight: 650;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      max-width: 62vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      height: calc(100vh - 52px);
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      min-height: 0;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }
    section {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(260px, 1.4fr) minmax(220px, 1fr);
    }
    .viewer {
      min-height: 0;
      position: relative;
      background: #0d1012;
      border-bottom: 1px solid var(--line);
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .overlay {
      position: absolute;
      left: 12px;
      bottom: 12px;
      color: var(--muted);
      background: rgba(14, 17, 20, .78);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      max-width: min(680px, calc(100% - 24px));
    }
    .images {
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--line);
    }
    .image-pane {
      min-width: 0;
      min-height: 0;
      background: var(--panel-2);
      display: grid;
      grid-template-rows: 34px minmax(0, 1fr);
    }
    .image-pane h2 {
      margin: 0;
      padding: 9px 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      border-bottom: 1px solid var(--line);
    }
    .image-pane img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      min-height: 0;
      background: #0d1012;
    }
    .group {
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .group:first-child { padding-top: 0; }
    label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 8px 0;
      color: var(--muted);
      font-size: 12px;
    }
    input[type="range"] { width: 100%; }
    input[type="number"] {
      width: 96px;
      background: #121619;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 6px 7px;
    }
    input[type="checkbox"] { transform: translateY(1px); }
    button {
      height: 34px;
      border: 1px solid var(--line);
      background: #273038;
      color: var(--text);
      border-radius: 5px;
      padding: 0 12px;
      font-weight: 620;
      cursor: pointer;
    }
    button:hover { border-color: #5c6973; }
    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 10px 0 4px;
    }
    .value {
      color: var(--text);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .meta strong { color: var(--text); font-weight: 620; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 42vh; }
      section { height: 58vh; }
      .images { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MDA Result Viewer</h1>
    <div id="resultPath" class="status">Loading scene...</div>
  </header>
  <main>
    <aside>
      <div class="group">
        <label>Frame <span id="frameLabel" class="value">0 / 0</span></label>
        <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" />
        <div class="buttons">
          <button id="playBtn">Play</button>
          <button id="reloadBtn">Reload 3D</button>
        </div>
        <label>FPS <input id="fpsInput" type="number" min="1" max="30" step="1" value="8" /></label>
        <label>Step <input id="stepInput" type="number" min="1" max="60" step="1" value="1" /></label>
      </div>
      <div class="group">
        <label>Max points <input id="maxPoints" type="number" min="1000" max="200000" step="1000" value="40000" /></label>
        <label>Stride <input id="strideInput" type="number" min="0" max="64" step="1" value="0" /></label>
        <label>Point size <input id="pointSize" type="number" min="1" max="12" step="0.5" value="2.5" /></label>
        <label>Focal scale <input id="focalScale" type="number" min="0.1" max="10" step="0.1" value="1.2" /></label>
      </div>
      <div class="group">
        <label>Depth min <input id="depthMin" type="number" step="0.1" value="0" /></label>
        <label>Depth max <input id="depthMax" type="number" step="0.1" value="1000000" /></label>
        <label><span>Filter depth edges</span><input id="edgeFilter" type="checkbox" checked /></label>
        <label>Edge rtol <input id="edgeRtol" type="number" min="0.001" max="1" step="0.005" value="0.03" /></label>
        <label><span>Mask black bg</span><input id="maskBlack" type="checkbox" /></label>
        <label><span>Mask white bg</span><input id="maskWhite" type="checkbox" /></label>
      </div>
      <div class="group meta" id="frameStats">Waiting for frame stats...</div>
      <div class="group meta" id="sceneStats">Waiting for scene metadata...</div>
    </aside>
    <section>
      <div class="viewer">
        <canvas id="gl"></canvas>
        <div class="overlay" id="cloudStats">Drag to rotate, wheel to zoom.</div>
      </div>
      <div class="images">
        <div class="image-pane">
          <h2>RGB</h2>
          <img id="rgbImage" alt="RGB frame" />
        </div>
        <div class="image-pane">
          <h2>Depth image</h2>
          <img id="depthImage" alt="Depth frame" />
        </div>
      </div>
    </section>
  </main>
  <script>
    const state = {
      frameCount: 0,
      frame: 0,
      playing: false,
      timer: null,
      requestId: 0,
      yaw: 0.55,
      pitch: 0.35,
      distance: 4.2,
      pointCount: 0,
    };

    const $ = (id) => document.getElementById(id);
    const canvas = $("gl");
    const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!gl) {
      $("cloudStats").textContent = "WebGL is unavailable in this browser.";
    }

    function shader(type, source) {
      const s = gl.createShader(type);
      gl.shaderSource(s, source);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }

    const program = gl ? (() => {
      const vs = shader(gl.VERTEX_SHADER, `
        attribute vec3 aPosition;
        attribute vec3 aColor;
        uniform mat4 uMvp;
        uniform float uPointSize;
        varying vec3 vColor;
        void main() {
          vColor = aColor / 255.0;
          gl_Position = uMvp * vec4(aPosition, 1.0);
          gl_PointSize = uPointSize;
        }
      `);
      const fs = shader(gl.FRAGMENT_SHADER, `
        precision mediump float;
        varying vec3 vColor;
        void main() {
          vec2 p = gl_PointCoord - vec2(0.5);
          if (dot(p, p) > 0.25) discard;
          gl_FragColor = vec4(vColor, 1.0);
        }
      `);
      const p = gl.createProgram();
      gl.attachShader(p, vs);
      gl.attachShader(p, fs);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    })() : null;

    const buffers = gl ? {
      pos: gl.createBuffer(),
      col: gl.createBuffer(),
      aPosition: gl.getAttribLocation(program, "aPosition"),
      aColor: gl.getAttribLocation(program, "aColor"),
      uMvp: gl.getUniformLocation(program, "uMvp"),
      uPointSize: gl.getUniformLocation(program, "uPointSize"),
    } : {};

    function mat4Perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0,
      ]);
    }
    function normalize(v) {
      const l = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / l, v[1] / l, v[2] / l];
    }
    function cross(a, b) {
      return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    }
    function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    function mat4LookAt(eye, center, up) {
      const z = normalize([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
      ]);
    }
    function mat4Mul(a, b) {
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
          out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
        }
      }
      return out;
    }
    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
      }
    }
    function render() {
      if (!gl || !program) return;
      resizeCanvas();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.05, 0.06, 0.07, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(program);
      const aspect = canvas.width / Math.max(canvas.height, 1);
      const proj = mat4Perspective(Math.PI / 4, aspect, 0.01, 100);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const eye = [state.distance * sy * cp, state.distance * sp, state.distance * cy * cp];
      const view = mat4LookAt(eye, [0,0,0], [0,1,0]);
      const mvp = mat4Mul(proj, view);
      gl.uniformMatrix4fv(buffers.uMvp, false, mvp);
      gl.uniform1f(buffers.uPointSize, Number($("pointSize").value));
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
      gl.enableVertexAttribArray(buffers.aPosition);
      gl.vertexAttribPointer(buffers.aPosition, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.col);
      gl.enableVertexAttribArray(buffers.aColor);
      gl.vertexAttribPointer(buffers.aColor, 3, gl.UNSIGNED_BYTE, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, state.pointCount);
    }
    function tick() {
      render();
      requestAnimationFrame(tick);
    }
    if (gl) tick();

    function b64ToBytes(b64) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes;
    }
    function uploadPointCloud(payload) {
      const points = new Float32Array(b64ToBytes(payload.points_b64).buffer);
      const colors = b64ToBytes(payload.colors_b64);
      const center = payload.center;
      const scale = payload.display_scale;
      for (let i = 0; i < points.length; i += 3) {
        points[i] = (points[i] - center[0]) * scale;
        points[i + 1] = (points[i + 1] - center[1]) * scale;
        points[i + 2] = (points[i + 2] - center[2]) * scale;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
      gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.col);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      state.pointCount = payload.count;
      const d = payload.depth_stats;
      $("cloudStats").textContent =
        `frame ${payload.frame} | points ${payload.count.toLocaleString()} | stride ${payload.stride} | ` +
        `depth min ${fmt(d.min)} max ${fmt(d.max)} mean ${fmt(d.mean)}`;
    }
    function fmt(v) {
      return (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(3);
    }
    async function loadScene() {
      const res = await fetch("/api/scene");
      const scene = await res.json();
      state.frameCount = scene.frame_count;
      $("frameSlider").max = Math.max(0, scene.frame_count - 1);
      $("resultPath").textContent = scene.result_dir;
      $("sceneStats").innerHTML =
        `<strong>${scene.frame_count}</strong> frames<br>` +
        `size <strong>${scene.width} x ${scene.height}</strong><br>` +
        `raw <strong>${scene.raw_count}</strong> depth <strong>${scene.depth_png_count}</strong><br>` +
        `depth range <strong>${fmt(scene.depth_min)} - ${fmt(scene.depth_max)}</strong><br>` +
        `first ${escapeHtml(scene.first_input)}<br>` +
        `last ${escapeHtml(scene.last_input)}`;
      await setFrame(0, true);
    }
    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    async function setFrame(frame, forceCloud=false) {
      frame = ((frame % state.frameCount) + state.frameCount) % state.frameCount;
      state.frame = frame;
      $("frameSlider").value = frame;
      $("frameLabel").textContent = `${frame} / ${state.frameCount - 1}`;
      $("rgbImage").src = `/api/frame/${frame}/rgb.png?t=${Date.now()}`;
      $("depthImage").src = `/api/frame/${frame}/depth.png?t=${Date.now()}`;
      fetch(`/api/frame/${frame}/stats`).then(r => r.json()).then(s => {
        $("frameStats").innerHTML =
          `input ${escapeHtml(s.input)}<br>` +
          `valid <strong>${s.valid_pixels.toLocaleString()}</strong><br>` +
          `depth min <strong>${fmt(s.depth_min)}</strong> max <strong>${fmt(s.depth_max)}</strong><br>` +
          `p01 <strong>${fmt(s.depth_p01)}</strong> p99 <strong>${fmt(s.depth_p99)}</strong>`;
      });
      if (forceCloud || state.playing) await loadPointCloud(frame);
    }
    async function loadPointCloud(frame = state.frame) {
      if (!gl) return;
      const id = ++state.requestId;
      const params = new URLSearchParams({
        max_points: $("maxPoints").value,
        stride: $("strideInput").value,
        depth_min: $("depthMin").value,
        depth_max: $("depthMax").value,
        focal_scale: $("focalScale").value,
        filter_depth_edges: $("edgeFilter").checked,
        depth_edge_rtol: $("edgeRtol").value,
        mask_black: $("maskBlack").checked,
        mask_white: $("maskWhite").checked,
      });
      $("cloudStats").textContent = `loading point cloud for frame ${frame}...`;
      const res = await fetch(`/api/frame/${frame}/pointcloud?${params.toString()}`);
      if (!res.ok) {
        $("cloudStats").textContent = `failed to load point cloud: ${res.status}`;
        return;
      }
      const payload = await res.json();
      if (id !== state.requestId) return;
      uploadPointCloud(payload);
    }
    function setPlaying(playing) {
      state.playing = playing;
      $("playBtn").textContent = playing ? "Pause" : "Play";
      if (state.timer) clearInterval(state.timer);
      state.timer = null;
      if (playing) {
        state.timer = setInterval(() => {
          const fps = Math.max(1, Number($("fpsInput").value) || 8);
          const step = Math.max(1, Number($("stepInput").value) || 1);
          setFrame(state.frame + step, true);
        }, 1000 / fps);
      }
    }

    $("frameSlider").addEventListener("input", (e) => setFrame(Number(e.target.value), true));
    $("playBtn").addEventListener("click", () => setPlaying(!state.playing));
    $("reloadBtn").addEventListener("click", () => loadPointCloud());
    $("pointSize").addEventListener("input", render);
    for (const id of ["maxPoints","strideInput","depthMin","depthMax","focalScale","edgeFilter","edgeRtol","maskBlack","maskWhite"]) {
      $(id).addEventListener("change", () => loadPointCloud());
    }

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointerup", () => { dragging = false; });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      state.yaw += dx * 0.006;
      state.pitch = Math.max(-1.35, Math.min(1.35, state.pitch + dy * 0.006));
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      state.distance = Math.max(1.2, Math.min(18, state.distance * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });

    loadScene().catch((err) => {
      $("resultPath").textContent = String(err);
      $("cloudStats").textContent = "Failed to load scene metadata.";
    });
  </script>
</body>
</html>
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser viewer for MDA result directories.")
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Directory containing frame_*.png and raw/*.npz, usually eval_results/.../<model_name>",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    scene = ResultScene.from_dir(args.result_dir)
    app = create_app(scene)
    print(f"Serving MDA result viewer for {scene.root}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
