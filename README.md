# weakly supervised edge detection

**CPS 843 / CP 8307: Computer Vision Final Project**

This repository contains the official PyTorch implementation for our final technical report. We investigate a weakly supervised approach to edge detection, utilizing a customized **Lightweight U-Net (480K parameters)** trained on noisy, algorithmically generated pseudo-labels from a modified Canny Edge Detector. 

Our central hypothesis is that a convolutional neural network can generalize beyond its flawed algorithmic supervisor, smoothing out high-frequency texture noise to better approximate human-consensus boundaries without requiring expensive human-annotated training data.

## 📊 Quantitative Results (BSDS500 Benchmark)

Evaluations were conducted on the benchmark Berkley Segmentation Dataset (BSDS500) strictly adhering to the official 200-image blind test split. A 1-pixel morphological tolerance window was applied to account for human annotator variance.

| Method / Contender | Frozen $\tau$* | Precision | Recall | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **1. Classical Canny (Algorithmic)** | N/A (Fixed) | 0.3928 | 0.4611 | 0.4242 |
| **2. U-Net (Trained on Raw Input)** | 0.70 | 0.3819 | 0.5176 | 0.4395 |
| **3. U-Net (Trained on Laplacian)** | 0.60 | 0.3982 | **0.5275** | **0.4538** |

*Note: The Laplacian-enhanced U-Net achieved a **14.4% relative gain** in recall compared to its classical supervisor.*

## ⚙️ Repository Structure

```text
├── dataset.py       # Custom PyTorch Dataset, Laplacian enhancement, pseudo-labels
├── model.py         # Lightweight U-Net (480K params) architecture
├── train.py         # Training script with Weighted BCE Loss class-imbalance handling
├── evaluate.py      # BSDS500 leak-free benchmark evaluation and threshold tuning
├── infer.py         # CLI tool for standalone edge detection on arbitrary images
├── visualize.py     # Qualitative Matplotlib visualization and comparison generator
