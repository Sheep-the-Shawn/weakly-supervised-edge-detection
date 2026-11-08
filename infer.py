# infer.py: Standalone Edge Detection Inference Tool
import argparse
import os
import cv2
import numpy as np
import torch
from dataset import apply_laplacian_enhancement
from model import LightweightUNet

def infer(image_path, checkpoint_path, output_path, use_laplacian=False, threshold=0.5, target_size=(256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
        
    orig_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 2. Preprocess
    if use_laplacian:
        model_input = apply_laplacian_enhancement(orig_resized)
    else:
        model_input = orig_resized
        
    input_tensor = (torch.from_numpy(model_input).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    
    # 3. Load model
    model = LightweightUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 4. Predict
    with torch.no_grad():
        prob_map = model.predict_probability(input_tensor).squeeze().cpu().numpy()
        
    binary_edges = ((prob_map > threshold) * 255).astype(np.uint8)
    
    # 5. Save output
    cv2.imwrite(output_path, binary_edges)
    print(f"Edge detection complete! Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run edge detection on an arbitrary image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (.jpg, .png)")
    parser.add_argument("--checkpoint", type=str, default="best_model_raw.pth", help="Path to model weights (.pth)")
    parser.add_argument("--output", type=str, default="predicted_edges.png", help="Path to save output edge image")
    parser.add_argument("--use_laplacian", action="store_true", help="Set this flag if the checkpoint was trained on Laplacian inputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Edge decision threshold in range [0, 1]")
    args = parser.parse_args()

    infer(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_laplacian=args.use_laplacian,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()