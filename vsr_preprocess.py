"""
VSR Preprocessing Pipeline
==========================
Generates three experimental versions of a 224x224 facial-crop video dataset:
  - Output_CLAHE        : Adaptive Contrast (CLAHE on grayscale, re-broadcast to BGR)
  - Output_Procrustes   : Similarity-transform alignment via MediaPipe FaceMesh
  - Output_PTM          : Partition-Time Masking spatiotemporal augmentation

Mirrors the input directory structure exactly. Non-mp4 sidecar files
(*.aac, *.txt, ...) are copied verbatim. Output mp4s are 224x224 uint8
H.264 video muxed with the ORIGINAL audio stream (copy) from the source mp4.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Lazy-imported in workers to avoid forking heavy state
_FACE_MESH = None


# ----------------------------- Constants ------------------------------------ #
FRAME_SIZE = 224
FPS_FALLBACK = 25.0

# CLAHE params (Sensors / MDPI 2025)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Procrustes canonical template (eyes horizontal, mouth at (112,160))
# Eye width chosen as 70px (typical inter-ocular distance for 224 face).
CANON_LEFT_EYE = np.array([77.0, 96.0], dtype=np.float32)   # subject's left = image right? we use image-coords
CANON_RIGHT_EYE = np.array([147.0, 96.0], dtype=np.float32)
CANON_MOUTH = np.array([112.0, 160.0], dtype=np.float32)

# MediaPipe FaceMesh landmark indices (468-pt topology)
LM_LEFT_EYE_CENTER = 468 - 1  # placeholder; replaced below
# Stable averaged eye-center indices: use eye-corner midpoints
LM_LEFT_EYE_CORNERS = (33, 133)   # outer, inner of subject's right eye in image
LM_RIGHT_EYE_CORNERS = (362, 263)
LM_MOUTH_CORNERS = (61, 291)

# PTM params (Sensors 2025)
PTM_GRID = 4               # 4x4 = 16 spatial partitions
PTM_ALPHA = 0.2            # temporal mask fraction
PTM_PARTS_PER_FRAME = 2    # partitions blacked out within masked frames


# --------------------------- Frame I/O helpers ------------------------------ #
def read_video_frames(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            fr = cv2.resize(fr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    return frames, float(fps)


def write_video_with_audio(frames: List[np.ndarray], fps: float,
                           src_mp4: Path, dst_mp4: Path) -> None:
    """Encode frames -> H.264 silent mp4, then mux with audio from src_mp4."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        silent = Path(td) / "v.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(silent), fourcc, fps, (FRAME_SIZE, FRAME_SIZE))
        if not vw.isOpened():
            raise IOError(f"Cannot open writer for {silent}")
        for f in frames:
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            vw.write(f)
        vw.release()

        # Mux: copy OpenCV's mp4v video stream as-is, attach source audio.
        # (Avoids libopenh264 re-encode which produced unplayable streams on
        # Fedora's ffmpeg-free build.)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent),
            "-i", str(src_mp4),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            str(dst_mp4),
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            # fall back: no audio, just copy the silent video
            shutil.copyfile(silent, dst_mp4)


# ---------------------------- Processing tracks ----------------------------- #
def apply_clahe(frames: List[np.ndarray]) -> List[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    out = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        eq = clahe.apply(gray)
        out.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
    return out


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        model_path = os.environ.get(
            "VSR_FACE_MODEL",
            str(Path(__file__).resolve().parent / "face_landmarker.task"),
        )
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _FACE_MESH = (vision.FaceLandmarker.create_from_options(opts), mp)
    return _FACE_MESH


def _landmarks_xy(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    fl, mp = _get_face_mesh()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = fl.detect(mp_img)
    if not res.face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    lms = res.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
    return pts


def _three_anchor_pts(pts: np.ndarray) -> np.ndarray:
    le = (pts[LM_LEFT_EYE_CORNERS[0]] + pts[LM_LEFT_EYE_CORNERS[1]]) / 2.0
    re = (pts[LM_RIGHT_EYE_CORNERS[0]] + pts[LM_RIGHT_EYE_CORNERS[1]]) / 2.0
    mo = (pts[LM_MOUTH_CORNERS[0]] + pts[LM_MOUTH_CORNERS[1]]) / 2.0
    return np.stack([le, re, mo]).astype(np.float32)


def apply_procrustes(frames: List[np.ndarray]) -> List[np.ndarray]:
    canon = np.stack([CANON_LEFT_EYE, CANON_RIGHT_EYE, CANON_MOUTH]).astype(np.float32)
    out: List[np.ndarray] = []
    last_M: Optional[np.ndarray] = None
    for f in frames:
        pts = _landmarks_xy(f)
        M: Optional[np.ndarray] = None
        if pts is not None:
            src = _three_anchor_pts(pts)
            M, _ = cv2.estimateAffinePartial2D(src, canon, method=cv2.LMEDS)
        if M is None:
            M = last_M
        if M is None:
            out.append(f.copy())
            continue
        warped = cv2.warpAffine(f, M, (FRAME_SIZE, FRAME_SIZE),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        out.append(warped)
        last_M = M
    return out


def apply_ptm(frames: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    T = len(frames)
    if T == 0:
        return frames
    tau = max(1, int(round(PTM_ALPHA * T)))
    tau = min(tau, T)
    start = int(rng.integers(0, T - tau + 1))
    masked_idx = set(range(start, start + tau))

    cell = FRAME_SIZE // PTM_GRID  # 56
    out: List[np.ndarray] = []
    for i, f in enumerate(frames):
        if i not in masked_idx:
            out.append(f)
            continue
        g = f.copy()
        choices = rng.choice(PTM_GRID * PTM_GRID, size=PTM_PARTS_PER_FRAME, replace=False)
        for c in choices:
            r, cc = divmod(int(c), PTM_GRID)
            y0, x0 = r * cell, cc * cell
            g[y0:y0 + cell, x0:x0 + cell] = 0
        out.append(g)
    return out


# ------------------------------ Orchestration ------------------------------- #
class VSRPreprocessor:
    def __init__(self, input_root: Path, out_clahe: Path,
                 out_procrustes: Path, out_ptm: Path,
                 tracks: Tuple[str, ...] = ("clahe", "procrustes", "ptm"),
                 seed: int = 0):
        self.input_root = input_root
        self.outs = {"clahe": out_clahe, "procrustes": out_procrustes, "ptm": out_ptm}
        self.tracks = tracks
        self.seed = seed

    def process_video(self, src: Path) -> str:
        rel = src.relative_to(self.input_root)
        try:
            frames, fps = read_video_frames(src)
        except Exception as e:
            return f"READ_FAIL {src}: {e}"
        if not frames:
            return f"EMPTY {src}"

        rng = np.random.default_rng(self.seed + abs(hash(str(rel))) % (2**31))

        for track in self.tracks:
            dst = self.outs[track] / rel
            if dst.exists():
                continue
            try:
                if track == "clahe":
                    out_frames = apply_clahe(frames)
                elif track == "procrustes":
                    out_frames = apply_procrustes(frames)
                elif track == "ptm":
                    out_frames = apply_ptm(frames, rng)
                else:
                    continue
                write_video_with_audio(out_frames, fps, src, dst)
            except Exception as e:
                return f"WRITE_FAIL[{track}] {src}: {e}"
        return f"OK {rel}"

    def copy_sidecar(self, src: Path) -> str:
        rel = src.relative_to(self.input_root)
        for track in self.tracks:
            dst = self.outs[track] / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                return f"COPY_FAIL {src}: {e}"
        return f"COPIED {rel}"


# Worker entry-points (must be top-level for multiprocessing pickling)
_WORKER: Optional[VSRPreprocessor] = None


def _init_worker(input_root: str, oc: str, op: str, opt: str,
                 tracks: Tuple[str, ...], seed: int):
    global _WORKER
    _WORKER = VSRPreprocessor(Path(input_root), Path(oc), Path(op), Path(opt),
                              tracks=tracks, seed=seed)


def _do_video(p: str) -> str:
    return _WORKER.process_video(Path(p))


def _do_copy(p: str) -> str:
    return _WORKER.copy_sidecar(Path(p))


def parallel_process(input_root: Path, out_clahe: Path, out_procrustes: Path,
                     out_ptm: Path, workers: int = 0,
                     tracks: Tuple[str, ...] = ("clahe", "procrustes", "ptm"),
                     seed: int = 0) -> None:
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    videos = sorted(input_root.rglob("*.mp4"))
    sidecars = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() != ".mp4"]

    print(f"[INFO] {len(videos)} videos / {len(sidecars)} sidecars / {workers} workers", flush=True)

    init_args = (str(input_root), str(out_clahe), str(out_procrustes), str(out_ptm), tracks, seed)

    # 1) sidecars: cheap, run in parallel first
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=init_args) as ex:
        for i, msg in enumerate(ex.map(_do_copy, [str(s) for s in sidecars], chunksize=64), 1):
            if msg.startswith(("COPY_FAIL", "FAIL")):
                print(msg, flush=True)
            if i % 5000 == 0:
                print(f"[sidecar] {i}/{len(sidecars)}", flush=True)

    # 2) videos
    done = 0
    fails = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=init_args) as ex:
        futs = [ex.submit(_do_video, str(v)) for v in videos]
        for fut in as_completed(futs):
            msg = fut.result()
            done += 1
            if not msg.startswith("OK"):
                fails += 1
                print(msg, flush=True)
            if done % 200 == 0:
                print(f"[video] {done}/{len(videos)}  fails={fails}", flush=True)
    print(f"[DONE] videos={done} fails={fails}", flush=True)


# --------------------------------- CLI -------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Baseline")
    ap.add_argument("--out-clahe", default="Output_CLAHE")
    ap.add_argument("--out-procrustes", default="Output_Procrustes")
    ap.add_argument("--out-ptm", default="Output_PTM")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tracks", default="clahe,procrustes,ptm",
                    help="comma list subset of clahe,procrustes,ptm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tracks = tuple(t.strip() for t in args.tracks.split(",") if t.strip())
    parallel_process(Path(args.input), Path(args.out_clahe),
                     Path(args.out_procrustes), Path(args.out_ptm),
                     workers=args.workers, tracks=tracks, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
