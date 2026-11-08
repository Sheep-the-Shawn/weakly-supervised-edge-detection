# dataset.py: BSDS500 PyTorch Dataset and Preprocessing Pipelines
import os
import glob
import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

def apply_laplacian_enhancement(img_gray, alpha=0.5):
    """
    Enhances high-frequency edge details using the 2nd-derivative Laplacian operator.
    Formula: Enhanced = Image - alpha * Laplacian(Image)
    """
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
    enhanced = img_gray.astype(np.float32) - (alpha * laplacian)
    return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

def generate_canny_pseudo_labels(img_gray, ksize=7, sigma=1.5, low_thresh=100, high_thresh=200):
    """
    Generates pseudo-ground-truth edge masks using classical Canny edge detection.
    Includes explicit Gaussian pre-smoothing to suppress high-frequency texture noise (grass, fur).
    """
    # Explicit Gaussian smoothing to filter macro-texture
    smoothed = cv2.GaussianBlur(img_gray, (ksize, ksize), sigmaX=sigma)
    edges = cv2.Canny(smoothed, threshold1=low_thresh, threshold2=high_thresh)
    return (edges > 0).astype(np.float32)

def load_human_ground_truth(mat_path, target_size=(256, 256)):
    """
    Parses BSDS500 MATLAB (.mat) ground truth annotations.
    Averages boundary maps across human subjects to form a consensus map.
    """
    mat = scipy.io.loadmat(mat_path)
    ground_truth = mat['groundTruth']
    num_annotators = ground_truth.shape[1]
    
    boundaries = [
        cv2.resize(ground_truth[0, i]['Boundaries'][0, 0].astype(np.float32), 
                   target_size, interpolation=cv2.INTER_NEAREST)
        for i in range(num_annotators)
    ]
    return np.mean(boundaries, axis=0).astype(np.float32)

def _find_default_data_dir():
    """Finds the BSDS500 data path regardless of clone or download structure."""
    candidates = [
        "./BSDS500_repo/BSDS500/data",
        "./BSDS500_repo/data",
        "./BSDS500_data/BSR/BSDS500/data",
        "./data"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "./BSDS500_repo/BSDS500/data"

class BSDS500Dataset(Dataset):
    def __init__(self, base_dir=None, split='train', 
                 use_laplacian=False, target_size=(256, 256)):
        """
        Args:
            base_dir: Path to BSDS500 'data' directory (auto-detected if None).
            split: 'train', 'val', or 'test'.
            use_laplacian: If True -> Exp 1 (Laplacian input). If False -> Exp 2 (Raw input).
            target_size: (Width, Height) tuple.
        """
        if base_dir is None:
            base_dir = _find_default_data_dir()
            
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        self.split = split
        self.use_laplacian = use_laplacian
        self.target_size = target_size
        
        self.img_paths = sorted(glob.glob(os.path.join(base_dir, "images", split, "*.jpg")))
        self.gt_paths = sorted(glob.glob(os.path.join(base_dir, "groundTruth", split, "*.mat")))
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found for split '{split}' in: {base_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Read grayscale image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize first to maintain crisp 1-pixel edges
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. Model input formulation (Raw vs Laplacian)
        model_input = apply_laplacian_enhancement(img_resized) if self.use_laplacian else img_resized
        
        # 4. Supervision target (Texture-smoothed Canny pseudo-label)
        canny_target = generate_canny_pseudo_labels(img_resized)
        
        # 5. Format to PyTorch tensors [1, H, W] in range [0.0, 1.0]
        input_tensor = torch.from_numpy(model_input).float().unsqueeze(0) / 255.0
        canny_tensor = torch.from_numpy(canny_target).float().unsqueeze(0)
        
        sample = {
            'input': input_tensor,
            'target': canny_tensor,
            'filename': os.path.basename(img_path)
        }
        
        # 6. Include human consensus ground truth during testing
        if self.split == 'test':
            sample['human_gt'] = torch.from_numpy(
                load_human_ground_truth(self.gt_paths[idx], self.target_size)
            ).float().unsqueeze(0)
            
        return sample