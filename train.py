# train.py: Training Script for Weakly Supervised Edge Detection
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BSDS500Dataset
from model import LightweightUNet

class WeightedEdgeLoss(nn.Module):
    def __init__(self, pos_weight_val=5.0, device='cpu'):
        super(WeightedEdgeLoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight_val]).to(device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        return self.bce(logits, targets)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train Edge Detection U-Net")
    parser.add_argument("--use_laplacian", action="store_true", help="Enhance inputs with Laplacian filter")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="best_model.pth", help="Checkpoint output filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Use Laplacian: {args.use_laplacian}")

    train_ds = BSDS500Dataset(split='train', use_laplacian=args.use_laplacian)
    val_ds = BSDS500Dataset(split='val', use_laplacian=args.use_laplacian)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LightweightUNet(in_channels=1, out_channels=1).to(device)
    criterion = WeightedEdgeLoss(pos_weight_val=5.0, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Val Loss: {val_loss:.4f} (Saved best)")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Val Loss: {val_loss:.4f}")

    print(f"Training finished. Best model saved to: {args.output}")

if __name__ == "__main__":
    main()