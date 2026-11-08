# visualize.py: Visual Comparison Tool for Report Figures
import argparse
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import (
    apply_laplacian_enhancement,
    generate_canny_pseudo_labels,
    load_human_ground_truth,
)
from model import LightweightUNet


def find_file(filename, search_dir="."):
  """Searches for an image or mat file anywhere in the project directory."""
  for root, _, files in os.walk(search_dir):
    if filename in files:
      return os.path.join(root, filename)
  return None


def main():
  parser = argparse.ArgumentParser(
      description="Visually compare edge detection methods on an image."
  )
  parser.add_argument(
      "--image",
      type=str,
      default="104010.jpg",
      help="Image filename (e.g. 104010.jpg) or full path",
  )
  parser.add_argument(
      "--tau_raw",
      type=float,
      default=0.70,
      help="Frozen threshold for Raw model",
  )
  parser.add_argument(
      "--tau_lap",
      type=float,
      default=0.60,
      help="Frozen threshold for Laplacian model",
  )
  args = parser.parse_args()

  # 1. Resolve image path
  img_path = (
      args.image if os.path.exists(args.image) else find_file(args.image)
  )
  if img_path is None:
    raise FileNotFoundError(
        f"Could not locate image '{args.image}' in project folders."
    )

  base_name = os.path.splitext(os.path.basename(img_path))[0]
  print(f"--> Processing: {img_path}")

  # 2. Load & standard resize
  raw_bgr = cv2.imread(img_path)
  gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
  gray_256 = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)

  # 3. Method 1: Classical Canny
  canny_edges = (generate_canny_pseudo_labels(gray_256) * 255).astype(np.uint8)

  # 4. Device and Model Loading
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_raw = LightweightUNet(1, 1).to(device)
  model_raw.load_state_dict(
      torch.load("best_model_raw.pth", map_location=device)
  )
  model_raw.eval()

  model_lap = LightweightUNet(1, 1).to(device)
  model_lap.load_state_dict(
      torch.load("best_model_laplacian.pth", map_location=device)
  )
  model_lap.eval()

  # 5. Method 2: U-Net (Raw Input)
  inp_raw = (
      torch.from_numpy(gray_256).float().unsqueeze(0).unsqueeze(0) / 255.0
  ).to(device)
  with torch.no_grad():
    prob_raw = (
        model_raw.predict_probability(inp_raw).squeeze().cpu().numpy()
    )
  edge_raw = ((prob_raw > args.tau_raw) * 255).astype(np.uint8)

  # 6. Method 3: U-Net (Laplacian Input)
  lap_img = apply_laplacian_enhancement(gray_256)
  inp_lap = (
      torch.from_numpy(lap_img).float().unsqueeze(0).unsqueeze(0) / 255.0
  ).to(device)
  with torch.no_grad():
    prob_lap = (
        model_lap.predict_probability(inp_lap).squeeze().cpu().numpy()
    )
  edge_lap = ((prob_lap > args.tau_lap) * 255).astype(np.uint8)

  # 7. Check for Human Ground Truth .mat file
  gt_path = find_file(f"{base_name}.mat")
  has_gt = gt_path is not None
  if has_gt:
    gt_map = load_human_ground_truth(gt_path, target_size=(256, 256))

  # 8. Plot Comparison Figure
  num_panels = 5 if has_gt else 4
  fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5))

  axes[0].imshow(gray_256, cmap="gray")
  axes[0].set_title(
      f"(a) Input Image\n[{base_name}.jpg]", fontsize=11, fontweight="bold"
  )
  axes[0].axis("off")

  axes[1].imshow(canny_edges, cmap="gray")
  axes[1].set_title(
      "(b) Classical Canny\n(Algorithmic)", fontsize=11, fontweight="bold"
  )
  axes[1].axis("off")

  axes[2].imshow(edge_raw, cmap="gray")
  axes[2].set_title(
      f"(c) U-Net (Raw)\n[tau* = {args.tau_raw:.2f}]",
      fontsize=11,
      fontweight="bold",
  )
  axes[2].axis("off")

  axes[3].imshow(edge_lap, cmap="gray")
  axes[3].set_title(
      f"(d) U-Net (Laplacian)\n[tau* = {args.tau_lap:.2f}]",
      fontsize=11,
      fontweight="bold",
  )
  axes[3].axis("off")

  if has_gt:
    axes[4].imshow(gt_map, cmap="magma")
    axes[4].set_title(
        "(e) Human Ground Truth\n(Consensus)", fontsize=11, fontweight="bold"
    )
    axes[4].axis("off")

  plt.tight_layout()
  output_filename = f"comparison_{base_name}.png"
  plt.savefig(output_filename, dpi=300, bbox_inches="tight")
  print(f">> Saved high-resolution comparison to '{output_filename}'")
  plt.show()


if __name__ == "__main__":
  main()