#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
League of Legends Q detector from YouTube VODs (color-cluster HUD method)

Pipeline
1) Download video from YouTube (or use local file).
2) Pass 1: scan the ROI over the whole video, cluster ROI mean colors (KMeans, k=4 by default).
   - Save swatches: cluster_colors/cluster_{i}_color.png  (BGR, 100x100)
   - Save example ROI frames per cluster: debug_rois/roi_{frame}_{mm-ss}_cluster{i}_example.png
   - Print centers so you can sanity-check.
3) Pass 2: pick the trigger cluster (CLI arg or auto by nearest-to-provided RGB), then
   iterate frames computing dominant color in the ROI and fire when ΔE2000 to the trigger
   cluster center is below a threshold. For each trigger:
   - Save ROI at trigger and +2s
   - Save a clip spanning [t-2s, t+3s] to clips/clip_{n}.mp4
   - Log events to events.csv

Notes
- ROI can be absolute pixels "x1,y1,x2,y2" or normalized "0.5,0.6,0.53,0.63".
- Colors are handled as BGR from OpenCV; ΔE is computed in Lab using skimage+colormath.
- Default trigger cluster is id=2 (often the “blue cast” state), but labels are arbitrary;
  pass --trigger-cluster explicitly or --trigger-rgb "R,G,B" to auto-pick closest center.
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color as skcolor
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# ----------------------------- utils ---------------------------------

def parse_roi(spec: str):
    """
    Accepts "x1,y1,x2,y2" where values are either ints (pixels) or floats in [0,1] (normalized).
    Returns tuple (x1,y1,x2,y2) along with a flag normalized.
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x1,y1,x2,y2'")
    vals = [float(p) for p in parts]
    is_norm = all(0 <= v <= 1 for v in vals)
    return vals, is_norm


def denorm_roi(vals, w, h):
    x1, y1, x2, y2 = vals
    return (int(round(x1 * w)), int(round(y1 * h)),
            int(round(x2 * w)), int(round(y2 * h)))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def rgb_to_lab(rgb):
    # rgb in [0..255]
    rgb_norm = np.array(rgb, dtype=np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb_norm[np.newaxis, np.newaxis, :])[0, 0]
    return lab


def delta_e2000_rgb(rgb1, rgb2):
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    c1 = LabColor(lab_l=float(lab1[0]), lab_a=float(lab1[1]), lab_b=float(lab1[2]))
    c2 = LabColor(lab_l=float(lab2[0]), lab_a=float(lab2[1]), lab_b=float(lab2[2]))
    return float(delta_e_cie2000(c1, c2))


def timecode(frame_idx, fps):
    total = frame_idx / max(fps, 1e-6)
    m = int(total // 60)
    s = int(total % 60)
    return f"{m:02d}:{s:02d}"


def best_stream_path(youtube_url: str, out_dir: Path, filename_stub: str) -> Path:
    """
    Download MP4 with pytube. You’ll need to `pip install pytube`.
    """
    try:
        from pytube import YouTube
    except Exception as e:
        print("ERROR: pytube is not installed. `pip install pytube`", file=sys.stderr)
        raise

    yt = YouTube(youtube_url)
    stream = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    if not stream:
        raise RuntimeError("No suitable MP4 progressive stream found.")
    out_dir = ensure_dir(out_dir)
    out_path = stream.download(output_path=str(out_dir), filename=f"{filename_stub}.mp4")
    return Path(out_path)


def frame_iter(cap):
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1


def extract_dominant_color(roi_bgr: np.ndarray, k=3):
    # Expect BGR uint8; convert to RGB for intuitions then back if needed
    pixels = roi_bgr.reshape(-1, 3).astype(np.float32)
    # KMeans in BGR space is fine for “dominant patch” purpose
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    # most frequent cluster
    dom_idx = np.argmax(np.bincount(labels))
    dom_bgr = centers[dom_idx]
    # return as RGB tuple for deltaE call convenience
    b, g, r = dom_bgr
    return np.array([r, g, b], dtype=np.float64), centers[:, ::-1]  # centers as RGB


# ----------------------------- main pipeline ---------------------------------

def pass1_cluster(video_path: Path,
                  roi_spec: str,
                  clusters_k: int,
                  sample_stride: int,
                  out_root: Path):
    """
    Return (centers_bgr [k,3], labels [N], roi_means_bgr [N,3])
    and save swatches + nearest-example ROIs.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vals, is_norm = parse_roi(roi_spec)
    if is_norm:
        x1, y1, x2, y2 = denorm_roi(vals, W, H)
    else:
        x1, y1, x2, y2 = map(int, vals)

    debug_dir = ensure_dir(out_root / "debug_rois")
    swatch_dir = ensure_dir(out_root / "cluster_colors")

    roi_means = []
    frame_indices = []

    t0 = time.time()
    for i, frame in frame_iter(cap):
        if sample_stride > 1 and (i % sample_stride) != 0:
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError("ROI is empty. Check coordinates and video resolution.")
        mean_bgr = roi.mean(axis=(0, 1))  # float64 BGR
        roi_means.append(mean_bgr)
        frame_indices.append(i)
    cap.release()

    roi_means = np.array(roi_means)
    if len(roi_means) < clusters_k:
        raise RuntimeError(f"Not enough samples for k={clusters_k}")

    print(f"[PASS1] samples={len(roi_means)}, fps≈{fps:.2f}, elapsed={time.time()-t0:.1f}s")

    # KMeans on mean colors
    km = KMeans(n_clusters=clusters_k, random_state=42)
    labels = km.fit_predict(roi_means)
    centers_bgr = km.cluster_centers_

    print("[PASS1] Cluster centers (BGR):")
    for i, c in enumerate(centers_bgr):
        print(f"  Cluster {i}: {c}")

    # Save swatches cluster_{i}_color.png (BGR)
    for i, c in enumerate(centers_bgr):
        patch = np.zeros((100, 100, 3), dtype=np.uint8)
        patch[:, :] = np.clip(c, 0, 255).astype(np.uint8)
        fn = swatch_dir / f"cluster_{i}_color.png"
        cv2.imwrite(str(fn), patch)

    # Save nearest example ROI image for each cluster
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps
    for cid in range(clusters_k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        pts = roi_means[idxs]
        d = np.linalg.norm(pts - centers_bgr[cid], axis=1)
        j = idxs[np.argmin(d)]
        frame_idx = frame_indices[j]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        roi = frame[y1:y2, x1:x2]
        tc = timecode(frame_idx, fps).replace(":", "-")
        out = debug_dir / f"roi_{frame_idx}_{tc}_cluster{cid}_example.png"
        cv2.imwrite(str(out), roi)
    cap.release()

    return centers_bgr, labels, np.array(frame_indices), roi_means


def pick_trigger_cluster(centers_bgr, args):
    """
    Decide trigger cluster id:
    - if --trigger-cluster given, use it.
    - elif --trigger-rgb R,G,B given, choose center closest in ΔE.
    - else default 2 (because humans love magical thinking).
    """
    if args.trigger_cluster is not None:
        cid = int(args.trigger_cluster)
        if cid < 0 or cid >= len(centers_bgr):
            raise ValueError("trigger_cluster out of range")
        return cid

    if args.trigger_rgb is not None:
        r, g, b = [float(x) for x in args.trigger_rgb.split(",")]
        # centers currently BGR; convert to RGB for ΔE
        centers_rgb = centers_bgr[:, ::-1]
        deltas = [delta_e2000_rgb([r, g, b], centers_rgb[i]) for i in range(len(centers_rgb))]
        cid = int(np.argmin(deltas))
        print(f"[PASS2] Auto-picked trigger cluster {cid} via ΔE to provided RGB; ΔE={deltas[cid]:.2f}")
        return cid

    print("[PASS2] No trigger specified, defaulting to cluster 2.")
    return 2


def write_clip(cap_path: Path, out_path: Path, start_frame: int, end_frame: int, fps: float, size_wh):
    w, h = size_wh
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap = cv2.VideoCapture(str(cap_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx <= end_frame:
        ret, fr = cap.read()
        if not ret:
            break
        out.write(fr)
        idx += 1
    out.release()
    cap.release()


def pass2_trigger(video_path: Path,
                  roi_spec: str,
                  trigger_center_bgr: np.ndarray,
                  out_root: Path,
                  delta_e_thresh: float,
                  dom_k: int,
                  pre_s: float,
                  post_s: float,
                  refractory_s: float,
                  status_every_s: float):
    clips_dir = ensure_dir(out_root / "clips")
    debug_dir = ensure_dir(out_root / "debug_rois")
    events_csv = out_root / "events.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vals, is_norm = parse_roi(roi_spec)
    if is_norm:
        x1, y1, x2, y2 = denorm_roi(vals, W, H)
    else:
        x1, y1, x2, y2 = map(int, vals)

    trig_rgb = trigger_center_bgr[::-1]
    last_fire_frame = -10**9
    refractory_frames = int(round(refractory_s * fps))
    pre_f = int(round(pre_s * fps))
    post_f = int(round(post_s * fps))
    status_every_f = max(1, int(round(status_every_s * fps)))

    width = W
    height = H

    clip_id = 0
    t0 = time.time()

    with open(events_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "frame", "timecode", "deltaE", "dom_R", "dom_G", "dom_B"])

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % status_every_f == 0:
                print(f"[STATUS] frame {idx} tc={timecode(idx, fps)}")

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                raise ValueError("ROI empty in pass2.")

            dom_rgb, _ = extract_dominant_color(roi, k=dom_k)
            de = delta_e2000_rgb(dom_rgb, trig_rgb)

            if de <= delta_e_thresh and (idx - last_fire_frame) >= refractory_frames:
                tc = timecode(idx, fps)
                print(f"[TRIGGER] frame {idx} tc={tc} ΔE={de:.2f} domRGB={tuple(dom_rgb.astype(int))}")

                # Save ROI at trigger and +2s
                tc_safe = tc.replace(":", "-")
                cv2.imwrite(str(debug_dir / f"roi_{idx}_{tc_safe}_trigger.png"), roi)

                plus2 = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, idx + int(round(2 * fps)))
                pos_now = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, plus2)
                ret2, frame2 = cap.read()
                if ret2:
                    roi2 = frame2[y1:y2, x1:x2]
                    cv2.imwrite(str(debug_dir / f"roi_{plus2}_{timecode(plus2, fps).replace(':','-')}_trigger_plus2s.png"), roi2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_now)

                # Clip
                start_f = max(0, idx - pre_f)
                end_f = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, idx + post_f)
                clip_path = clips_dir / f"clip_{clip_id}.mp4"
                write_clip(video_path, clip_path, start_f, end_f, fps, (width, height))
                wr.writerow([clip_id, idx, tc, f"{de:.3f}", int(dom_rgb[0]), int(dom_rgb[1]), int(dom_rgb[2])])
                clip_id += 1
                last_fire_frame = idx

            idx += 1

    cap.release()
    print(f"[PASS2] Done in {time.time()-t0:.1f}s. Events CSV: {events_csv}")


def main():
    ap = argparse.ArgumentParser(description="League of Legends Q ability detector via ROI color clusters")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube", type=str, help="YouTube URL to download")
    src.add_argument("--video", type=str, help="Local MP4 path")

    ap.add_argument("--roi", type=str, default="540,662,560,685",
                    help="ROI 'x1,y1,x2,y2' in px or normalized 0..1 (e.g. '0.5,0.61,0.52,0.635')")
    ap.add_argument("--clusters", type=int, default=4, help="K for pass1 clustering")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride for pass1 sampling (1=every frame)")
    ap.add_argument("--out", type=str, default="outputs/run1", help="Output root directory")

    ap.add_argument("--trigger-cluster", type=int, help="Cluster id to treat as trigger (from pass1)")
    ap.add_argument("--trigger-rgb", type=str, help="RGB like '95,70,39' to auto-pick closest center")
    ap.add_argument("--deltaE", type=float, default=10.0, help="ΔE2000 threshold to fire")
    ap.add_argument("--dom-k", type=int, default=3, help="K for dominant-color KMeans in pass2")
    ap.add_argument("--pre", type=float, default=2.0, help="Clip seconds before trigger")
    ap.add_argument("--post", type=float, default=3.0, help="Clip seconds after trigger")
    ap.add_argument("--refractory", type=float, default=2.0, help="Seconds to suppress duplicate triggers")
    ap.add_argument("--status-every", type=float, default=5.0, help="Status print interval (seconds)")

    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    # Acquire video path
    if args.youtube:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.youtube)[:40]
        video_path = best_stream_path(args.youtube, out_root, f"video_{slug}")
    else:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

    # Pass 1
    centers_bgr, labels, frame_idxs, roi_means = pass1_cluster(
        video_path=video_path,
        roi_spec=args.roi,
        clusters_k=args.clusters,
        sample_stride=args.stride,
        out_root=out_root
    )

    # Decide trigger cluster
    trig_cid = pick_trigger_cluster(centers_bgr, args)
    trig_center_bgr = centers_bgr[trig_cid]
    print(f"[PASS2] Using trigger cluster {trig_cid} center(BGR)={trig_center_bgr} center(RGB)={trig_center_bgr[::-1]}")

    # Pass 2
    pass2_trigger(
        video_path=video_path,
        roi_spec=args.roi,
        trigger_center_bgr=trig_center_bgr,
        out_root=out_root,
        delta_e_thresh=args.deltaE,
        dom_k=args.dom_k,
        pre_s=args.pre,
        post_s=args.post,
        refractory_s=args.refractory,
        status_every_s=args.status_every
    )


if __name__ == "__main__":
    main()
