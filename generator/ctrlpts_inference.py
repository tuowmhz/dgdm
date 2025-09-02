#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn

# --- Repo-local imports (match train.py paths) ---
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# generator / dynamics / assets modules
from generator.diffusion import Diffusion
from generator.diffusion_utils import ConditionalUnet1D
from dynamics.profile_forward_2d import ProfileForward2DModel
from assets.icon_process import extract_contours

# --------------------------
# CONFIG — EDIT ME
# --------------------------
OBJECT_NPY = "/absolute/path/to/refined_mask.npy"          # <-- npy only
DIFFUSION_CKPT_PATH = "/absolute/path/to/diffusion.ckpt"
CLASSIFIER_CKPT_PATH = "/absolute/path/to/classifier.pth"

# Model/guide hyper-params: must match training
NUM_TRAIN_TIMESTEPS   = 1000
NUM_INFERENCE_STEPS   = 50
CTRLPTS_DIM           = 14     # 2D case: 7 per finger => 14 total
INPUT_SPLINE_DIM      = 1      # y-displacement only in 2D
DOWN_DIMS             = [128, 256]
DIFF_EMBED_DIM        = 32

# Classifier was trained with a fixed number of object vertices.
# Set this to the SAME value used during training (commonly 512, 1024, etc.).
OBJECT_MAX_NUM_VERTICES = 512

# Convergence/grid sampling knobs (used inside cond_fn); match training if you changed them.
GRID_SIZE = 11
NUM_POS   = 7

# Normalization bounds for 2D objects (MUST match train.py)
OBJ_MIN_X, OBJ_MAX_X = -0.05, 0.05
OBJ_MIN_Y, OBJ_MAX_Y = -0.05, 0.05

# Normalization bounds for 2D control-point y (only used if you denormalize CSV)
CP_MIN_Y, CP_MAX_Y   = -0.045, 0.015

# Output
OUT_DIR   = "./inference_outputs"
OUT_CSV   = os.path.join(OUT_DIR, "convergence_control_points.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0
BATCH_SIZE = 1  # number of gripper candidates to generate
USE_EMA = True  # prefer EMA weights if available

# --------------------------
# Utilities
# --------------------------

def load_npy_image(path: str) -> np.ndarray:
    """
    STRICTLY loads a .npy file and returns an HWC uint8 image for extract_contours().
    Accepted shapes:
      - dict-like npy with key 'image' (NCHW or CHW). Uses the first image if NCHW.
      - CHW -> converted to HWC
      - NCHW -> first item -> HWC
      - HxW (mask) -> stacked to HxWx3
      - HxWx{1,3,4} -> used as-is (channels last)
    """
    if not path.lower().endswith(".npy"):
        raise ValueError("This script only accepts .npy inputs. Please provide a .npy file.")

    arr = np.load(path, allow_pickle=True)

    # Handle dict-like npy saved via np.save(obj) where obj is a dict
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            data = arr.item()
        except Exception as e:
            raise ValueError(f".npy object array is not a dict-like structure: {e}")
        if "image" not in data:
            raise ValueError("Expected key 'image' in the .npy dict.")
        images = data["image"]
    else:
        # Raw array cases
        images = arr

    # Now normalize to HWC
    if images.ndim == 4:
        # NCHW -> take first N
        img = images[0].transpose(1, 2, 0)
    elif images.ndim == 3:
        # Could be CHW or HWC. Heuristic: if first dim is small (<=4), assume CHW.
        if images.shape[0] <= 4 and images.shape[2] > 4:
            img = images.transpose(1, 2, 0)  # CHW -> HWC
        else:
            img = images  # HWC already
    elif images.ndim == 2:
        # Grayscale mask HxW -> HxWx3
        img = np.stack([images, images, images], axis=-1)
    else:
        raise ValueError(f"Unsupported array shape for image: {images.shape}")

    # Ensure uint8 HWC
    if img.dtype != np.uint8:
        img = (255.0 * (img - img.min()) / (img.ptp() + 1e-8)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def resample_contour(points_xy: np.ndarray, target_n: int) -> np.ndarray:
    """
    Resample a 2D polyline 'points_xy' (V,2) to exactly 'target_n' vertices.
    Uses arc-length parameterization + linear interpolation.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] == target_n:
        return pts

    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1) if diffs.size else np.array([0.0], dtype=np.float32)
    arclen = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(arclen[-1])
    if total < 1e-8:
        return np.repeat(pts[:1], target_n, axis=0)

    t_old = arclen / total
    t_new = np.linspace(0.0, 1.0, target_n)

    x_new = np.interp(t_new, t_old, pts[:, 0])
    y_new = np.interp(t_new, t_old, pts[:, 1])
    return np.stack([x_new, y_new], axis=-1).astype(np.float32)


def normalize_xy(points_xy: np.ndarray,
                 min_x: float, max_x: float,
                 min_y: float, max_y: float) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x = (x - min_x) / (max_x - min_x) * 2.0 - 1.0
    y = (y - min_y) / (max_y - min_y) * 2.0 - 1.0
    return np.stack([x, y], axis=-1).astype(np.float32)


def save_cp_csv(cp_final: np.ndarray, csv_path: str,
                cp_min_y: float = CP_MIN_Y, cp_max_y: float = CP_MAX_Y,
                denormalize: bool = False):
    """
    cp_final: shape (num_points, 1) normalized in [-1,1] (model space)
    Exports a 3-column CSV: index, left_y, right_y
    - left finger = first half (indices 0..N-1)
    - right finger = second half (indices N..2N-1)
    """
    num_points = cp_final.shape[0]
    assert num_points % 2 == 0, "Expected even number of control points (left+right)."
    n_per_finger = num_points // 2

    left = cp_final[:n_per_finger, 0]
    right = cp_final[n_per_finger:, 0]

    if denormalize:
        # Map back from [-1,1] -> [CP_MIN_Y, CP_MAX_Y]
        left  = (left + 1.0) * 0.5 * (cp_max_y - cp_min_y) + cp_min_y
        right = (right + 1.0) * 0.5 * (cp_max_y - cp_min_y) + cp_min_y

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "left_y", "right_y"])
        for i in range(n_per_finger):
            writer.writerow([i, float(left[i]), float(right[i])])


# --------------------------
# Main
# --------------------------

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) Load the object npy and extract a contour
    img_hwc = load_npy_image(OBJECT_NPY)             # HWC uint8
    contour_xy = extract_contours(img_hwc)           # (V,2) in image coords
    if not isinstance(contour_xy, np.ndarray):
        contour_xy = np.asarray(contour_xy, dtype=np.float32)
    contour_xy = contour_xy.astype(np.float32)

    # 2) Resample to the classifier’s expected vertex count and normalize to [-1,1]
    contour_xy = resample_contour(contour_xy, OBJECT_MAX_NUM_VERTICES)
    contour_xy = normalize_xy(contour_xy, OBJ_MIN_X, OBJ_MAX_X, OBJ_MIN_Y, OBJ_MAX_Y)

    # 3) Build object_vertices tensor like train.py (batch of one object)
    object_vertices = torch.from_numpy(contour_xy).float().unsqueeze(0)  # (1, V, 2)
    object_ids = [0]  # dummy id for this single object

    # 4) Build models & scheduler (match train.py)
    scheduler = DDIMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    unet = ConditionalUnet1D(
        input_dim=INPUT_SPLINE_DIM,
        global_cond_dim=0,
        down_dims=DOWN_DIMS,
        diffusion_step_embed_dim=DIFF_EMBED_DIM,
    )

    classifier_core = ProfileForward2DModel(
        output_ch=3,
        params_ch=CTRLPTS_DIM,
        object_ch=2 * OBJECT_MAX_NUM_VERTICES
    )
    if torch.cuda.is_available():
        classifier_model = nn.DataParallel(classifier_core.to(DEVICE))
    else:
        classifier_model = classifier_core.to(DEVICE)
    # Load classifier weights (same loader style as train.py)
    print("Loading classifier checkpoint:", CLASSIFIER_CKPT_PATH)
    classifier_state = torch.load(CLASSIFIER_CKPT_PATH, map_location=DEVICE)
    # Allow both plain state_dict and {'state_dict': ...}
    if isinstance(classifier_state, dict) and "state_dict" in classifier_state:
        classifier_model.load_state_dict(classifier_state["state_dict"])
    else:
        classifier_model.load_state_dict(classifier_state)
    classifier_model.eval()
    for p in classifier_model.parameters():
        p.requires_grad = False

    # 5) Assemble Diffusion module
    diffusion_model = Diffusion(
        noise_pred_net=unet,
        noise_scheduler=scheduler,
        num_inference_steps=NUM_INFERENCE_STEPS,
        mode="point",
        input_dim=INPUT_SPLINE_DIM,
        num_points=CTRLPTS_DIM,
        learning_rate=1e-4,            # unused for inference
        lr_warmup_steps=0,             # unused for inference
        ema_power=0.0,                 # unused for inference
        class_cond=True,
        classifier_model=classifier_model,
        grid_size=GRID_SIZE,
        num_pos=NUM_POS,
        object_vertices=object_vertices,  # our single object
        object_ids=object_ids,
        num_cpus=4,
        pts_x_dim=7,                      # irrelevant in 2D path but required by ctor
        pts_z_dim=1,                      # irrelevant in 2D
        sub_batch_size=1,
        render_video=False,
        seed=SEED,
    ).to(DEVICE)
    diffusion_model.eval()

    # 6) Load diffusion checkpoint (Lightning: use 'state_dict' if present)
    print("Loading diffusion checkpoint:", DIFFUSION_CKPT_PATH)
    ckpt = torch.load(DIFFUSION_CKPT_PATH, map_location=DEVICE)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    diffusion_model.load_state_dict(state_dict, strict=False)

    # 7) Run **convergence-only** guided inference and obtain FINAL control points
    results = diffusion_model.infer_convergence_control_points(
        batch_size=BATCH_SIZE,
        ori_range=[-1.0, 1.0],
        save_dir=None,            # we export CSV below
        seed=SEED,
        return_numpy=True,
        use_ema_weights=USE_EMA,
    )
    # 'results' is a list over objects; we passed 1 object, so results[0] is (B, num_points, 1)
    cp = results[0][0]  # (num_points, 1)

    # 8) Save to CSV (index, left_y, right_y). Choose denormalize=True if you want meters back.
    os.makedirs(OUT_DIR, exist_ok=True)
    save_cp_csv(cp, OUT_CSV, denormalize=False)
    print(f"Saved control points CSV to: {OUT_CSV}")


if __name__ == "__main__":
    # Ensure repo root (so relative imports resolve if you run from elsewhere)
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    main()