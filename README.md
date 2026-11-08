# weakly supervised edge detection

**CPS 843 / CP 8307: Computer Vision Final Project**

This repository contains the official PyTorch implementation for our final technical report. We investigate a weakly supervised approach to edge detection, utilizing a customized **Lightweight U-Net (480K parameters)** trained on noisy, algorithmically generated pseudo-labels from a modified Canny Edge Detector. 

Our central hypothesis is that a convolutional neural network can generalize beyond its flawed algorithmic supervisor, smoothing out high-frequency texture noise to better approximate human-consensus boundaries without requiring expensive human-annotated training data.

## Quantitative Results (BSDS500 Benchmark)

Evaluations were conducted on the benchmark Berkley Segmentation Dataset (BSDS500) strictly adhering to the official 200-image blind test split. A 1-pixel morphological tolerance window was applied to account for human annotator variance.

| Method / Contender | Frozen $\tau$* | Precision | Recall | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **1. Classical Canny (Algorithmic)** | N/A (Fixed) | 0.3928 | 0.4611 | 0.4242 |
| **2. U-Net (Trained on Raw Input)** | 0.70 | 0.3819 | 0.5176 | 0.4395 |
| **3. U-Net (Trained on Laplacian)** | 0.60 | 0.3982 | **0.5275** | **0.4538** |

*Note: The Laplacian-enhanced U-Net achieved a **14.4% relative gain** in recall compared to its classical supervisor.*

## Repository Structure

```text
├── dataset.py       # Custom PyTorch Dataset, Laplacian enhancement, pseudo-labels
├── model.py         # Lightweight U-Net (480K params) architecture
├── train.py         # Training script with Weighted BCE Loss class-imbalance handling
├── evaluate.py      # BSDS500 leak-free benchmark evaluation and threshold tuning
├── infer.py         # CLI tool for standalone edge detection on arbitrary images
├── visualize.py     # Qualitative Matplotlib visualization and comparison generator
```

## 🚀 Installation & Setup

> **Note:** Run the following commands in your terminal (Command Prompt / PowerShell on Windows, or Terminal on macOS/Linux) while located in the project's root folder.

### 1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```
*(If you downloaded the repository as a ZIP from GitHub, simply extract it and open your terminal inside the extracted folder).*

### 2. Install dependencies:
```
pip install -r requirements.txt
```

### 3. Dataset Preparation:
Download the [BSDS500 Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) and place the dataset's `data` folder inside this project's root directory. The `dataset.py` loader is pre-programmed to automatically locate folders named `data/` or `BSDS500_repo/`.

---

## 💻 Reproducibility & Usage

All commands below should be executed from the project root directory.

### 1. Training the Models
The networks are optimized using a Weighted Binary Cross-Entropy Loss ($\beta = 5.0$) to counteract the ≈ 10:1 background-to-edge class imbalance.

* **Train the Raw Image U-Net (Experiment 1):**
```
python train.py --epochs 20 --batch_size 8 --lr 1e-3 --output best_model_raw.pth
```

* **Train the Laplacian-Enhanced U-Net (Experiment 2):**
```
python train.py --use_laplacian --epochs 20 --batch_size 8 --lr 1e-3 --output best_model_laplacian.pth
```

### 2. Evaluating on the BSDS500 Benchmark
This script automatically executes our leak-free two-phase pipeline:
1. **Phase 1:** Calibrates the optimal decision threshold ($\tau^*$) using the 100-image validation split.
2. **Phase 2:** Freezes $\tau^*$ and evaluates final Precision, Recall, and F1-score against human consensus on the held-out 200-image test set.

```
python evaluate.py
```

### 3. Qualitative Visualizations
Generate a full side-by-side comparison figure (Original Image, Classical Canny, Raw U-Net, Laplacian U-Net, and Human Consensus Ground Truth) for any test image:
```
python visualize.py --image 104010.jpg
```
*(The high-resolution comparison figure is automatically generated and saved to the project folder).*

### 4. Running Standalone Inference on Arbitrary Images
You can run edge detection on any custom image using our standalone inference tool:

* **Using the Raw Input checkpoint:**
```
python infer.py --image your_image.jpg --checkpoint best_model_raw.pth --output predicted_edges.png --threshold 0.70
```

* **Using the Laplacian-Enhanced checkpoint:**
```
python infer.py --image your_image.jpg --checkpoint best_model_laplacian.pth --use_laplacian --output predicted_edges.png --threshold 0.60
```

---

## 🧠 Technical Highlights

* **Lightweight U-Net (480K parameters):** Standard segmentation U-Nets typically have over 30M parameters, which easily overfit to noise. Our custom 480K parameter design acts as an information bottleneck, compelling the network to learn global boundary structures rather than memorizing Canny micro-texture artifacts.
* **Laplacian Induction ($\alpha = 0.5$):** Pre-sharpening inputs using a discrete 2nd-derivative Laplacian kernel introduces an inductive bias that amplifies gradient discontinuities, streamlining boundary convergence.
* **Weighted Class Objective:** By weighting positive loss contributions by $\beta = 5.0$, we prevent trivial convergence to all-background predictions while maintaining edge precision.

---

## 🔗 Acknowledgements
* **Benchmark:** The Berkeley Segmentation Dataset and Benchmark (BSDS500).
* **Course:** CPS 843 / CP 8307: Computer Vision, Department of Computer Science.
