# evaluate.py: Rigorous Leak-Free BSDS500 Benchmark against Human Ground Truth
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    BSDS500Dataset,
    generate_canny_pseudo_labels,
    load_human_ground_truth,
)
from model import LightweightUNet


def compute_edge_metrics(pred_binary, gt_binary, tolerance=1):
  """Computes Precision, Recall, and F1 components with +/- 1-pixel spatial slack

  to account for human annotator variances.
  """
  pred_bool = pred_binary.astype(bool)
  gt_bool = gt_binary.astype(bool)

  if tolerance > 0:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_dilated = cv2.dilate(gt_binary.astype(np.uint8), kernel).astype(bool)
    pred_dilated = cv2.dilate(pred_binary.astype(np.uint8), kernel).astype(bool)

    tp_precision = np.logical_and(pred_bool, gt_dilated).sum()
    fp = np.logical_and(pred_bool, ~gt_dilated).sum()
    tp_recall = np.logical_and(gt_bool, pred_dilated).sum()
    fn = np.logical_and(gt_bool, ~pred_dilated).sum()
  else:
    tp_precision = np.logical_and(pred_bool, gt_bool).sum()
    tp_recall = tp_precision
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

  return tp_precision, fp, tp_recall, fn


def evaluate_at_threshold(probs, gts, threshold):
  """Evaluates dataset-wide Precision, Recall, and F1 at a single fixed threshold."""
  tot_tp_p, tot_fp, tot_tp_r, tot_fn = 0, 0, 0, 0
  for p_map, gt in zip(probs, gts):
    pred = (p_map > threshold).astype(np.float32)
    tp_p, fp, tp_r, fn = compute_edge_metrics(pred, gt)
    tot_tp_p += tp_p
    tot_fp += fp
    tot_tp_r += tp_r
    tot_fn += fn

  prec = tot_tp_p / (tot_tp_p + tot_fp + 1e-8)
  rec = tot_tp_r / (tot_tp_r + tot_fn + 1e-8)
  f1 = (2 * prec * rec) / (prec + rec + 1e-8)
  return prec, rec, f1


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Running evaluation on device: {device}")

  # 1. Load Trained Checkpoints
  model_raw = LightweightUNet(in_channels=1, out_channels=1).to(device)
  model_raw.load_state_dict(
      torch.load("best_model_raw.pth", map_location=device)
  )
  model_raw.eval()

  model_lap = LightweightUNet(in_channels=1, out_channels=1).to(device)
  model_lap.load_state_dict(
      torch.load("best_model_laplacian.pth", map_location=device)
  )
  model_lap.eval()

  # --------------------------------------------------------------------------
  # PHASE 1: TUNE THRESHOLD ON VALIDATION SET (100 IMAGES)
  # --------------------------------------------------------------------------
  print("\n[Phase 1] Tuning optimal decision thresholds on Validation set...")
  val_ds_raw = BSDS500Dataset(split="val", use_laplacian=False)
  val_ds_lap = BSDS500Dataset(split="val", use_laplacian=True)

  val_loader_raw = DataLoader(val_ds_raw, batch_size=1, shuffle=False)
  val_loader_lap = DataLoader(val_ds_lap, batch_size=1, shuffle=False)

  val_probs_raw, val_probs_lap, val_gts = [], [], []

  with torch.no_grad():
    for i, (sample_raw, sample_lap) in enumerate(
        zip(val_loader_raw, val_loader_lap)
    ):
      gt = (load_human_ground_truth(val_ds_raw.gt_paths[i]) > 0.2).astype(
          np.float32
      )
      val_gts.append(gt)

      inp_r = sample_raw["input"].to(device)
      val_probs_raw.append(
          model_raw.predict_probability(inp_r).squeeze().cpu().numpy()
      )

      inp_l = sample_lap["input"].to(device)
      val_probs_lap.append(
          model_lap.predict_probability(inp_l).squeeze().cpu().numpy()
      )

  candidate_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

  def find_best_tau(probs, gts, name):
    best_t, best_f1 = 0.5, -1.0
    for t in candidate_thresholds:
      _, _, f1 = evaluate_at_threshold(probs, gts, t)
      if f1 > best_f1:
        best_f1, best_t = f1, t
    print(
        f"  --> Optimal {name} Threshold: tau* = {best_t:.2f} (Val F1:"
        f" {best_f1:.4f})"
    )
    return best_t

  tau_raw = find_best_tau(val_probs_raw, val_gts, "U-Net Raw")
  tau_lap = find_best_tau(val_probs_lap, val_gts, "U-Net Laplacian")

  # --------------------------------------------------------------------------
  # PHASE 2: EVALUATE ON UNSEEN TEST SET (200 IMAGES) WITH FROZEN TAU*
  # --------------------------------------------------------------------------
  print("\n[Phase 2] Evaluating 200 unseen Test images with frozen tau*...")
  test_ds_raw = BSDS500Dataset(split="test", use_laplacian=False)
  test_ds_lap = BSDS500Dataset(split="test", use_laplacian=True)

  test_loader_raw = DataLoader(test_ds_raw, batch_size=1, shuffle=False)
  test_loader_lap = DataLoader(test_ds_lap, batch_size=1, shuffle=False)

  test_probs_raw, test_probs_lap, test_gts, raw_images_u8 = [], [], [], []

  with torch.no_grad():
    for i, (sample_raw, sample_lap) in enumerate(
        zip(test_loader_raw, test_loader_lap)
    ):
      gt = (load_human_ground_truth(test_ds_raw.gt_paths[i]) > 0.2).astype(
          np.float32
      )
      test_gts.append(gt)

      raw_u8 = (sample_raw["input"].squeeze().numpy() * 255.0).astype(np.uint8)
      raw_images_u8.append(raw_u8)

      inp_r = sample_raw["input"].to(device)
      test_probs_raw.append(
          model_raw.predict_probability(inp_r).squeeze().cpu().numpy()
      )

      inp_l = sample_lap["input"].to(device)
      test_probs_lap.append(
        model_lap.predict_probability(inp_l).squeeze().cpu().numpy()
    )

  # 1. Classical Canny Baseline
  c_tp_p, c_fp, c_tp_r, c_fn = 0, 0, 0, 0
  for raw_img, gt in zip(raw_images_u8, test_gts):
    canny_map = generate_canny_pseudo_labels(raw_img)
    tp_p, fp, tp_r, fn = compute_edge_metrics(canny_map, gt)
    c_tp_p += tp_p
    c_fp += fp
    c_tp_r += tp_r
    c_fn += fn

  canny_p = c_tp_p / (c_tp_p + c_fp + 1e-8)
  canny_r = c_tp_r / (c_tp_r + c_fn + 1e-8)
  canny_f1 = (2 * canny_p * canny_r) / (canny_p + canny_r + 1e-8)

  # 2. Frozen Test Evaluation for Models
  p_raw, r_raw, f1_raw = evaluate_at_threshold(test_probs_raw, test_gts, tau_raw)
  p_lap, r_lap, f1_lap = evaluate_at_threshold(test_probs_lap, test_gts, tau_lap)

  # --------------------------------------------------------------------------
  # PRINT FINAL RESULTS TABLE
  # --------------------------------------------------------------------------
  print("\n" + "=" * 84)
  print(
      "   BSDS500 FINAL BENCHMARK (TUNED ON 100 VAL, EVALUATED ON 200 UNSEEN"
      " TEST)"
  )
  print("=" * 84)
  print(
      f"{'Method / Contender':<28} | {'Frozen tau*':<12} | {'Precision':<10} |"
      f" {'Recall':<10} | {'Test F1-Score':<12}"
  )
  print("-" * 84)
  print(
      f"{'1. Classical Canny':<28} | {'N/A (Fixed)':<12} | {canny_p:<10.4f} |"
      f" {canny_r:<10.4f} | {canny_f1:<12.4f}"
  )
  print(
      f"{'2. U-Net (Trained on Raw)':<28} | {tau_raw:<12.2f} | {p_raw:<10.4f} |"
      f" {r_raw:<10.4f} | {f1_raw:<12.4f}"
  )
  print(
      f"{'3. U-Net (Trained on Lap)':<28} | {tau_lap:<12.2f} | {p_lap:<10.4f} |"
      f" {r_lap:<10.4f} | {f1_lap:<12.4f}"
  )
  print("=" * 84)


if __name__ == "__main__":
  main()