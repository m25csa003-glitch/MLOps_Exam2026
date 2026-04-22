import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from model import UNet

NUM_CLASSES = 23
EPOCHS      = 15
LR          = 1e-3
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(preds, masks, num_classes=NUM_CLASSES):
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    iou_list, dice_list = [], []
    for cls in range(num_classes):
        p = (preds == cls)
        m = (masks == cls)
        inter = (p & m).sum()
        union = (p | m).sum()
        denom = p.sum() + m.sum()
        iou_list.append(inter / (union + 1e-6) if union else 1.0)
        dice_list.append(2*inter / (denom + 1e-6) if denom else 1.0)
    return np.mean(iou_list), np.mean(dice_list)


def train():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, _, _, _ = get_dataloaders(
        data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
        batch_size=BATCH_SIZE
    )

    model     = UNet(3, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"loss": [], "miou": [], "mdice": []}

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Train metrics
        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                iou, dice = compute_metrics(preds, masks)
                ious.append(iou); dices.append(dice)

        miou  = np.mean(ious)
        mdice = np.mean(dices)
        history["loss"].append(avg_loss)
        history["miou"].append(miou)
        history["mdice"].append(mdice)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} | mIOU={miou:.4f} | mDice={mdice:.4f}")
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "unet_cityscape.pth"))
    print("Model saved!")

    # Test metrics
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            iou, dice = compute_metrics(preds, masks)
            ious.append(iou); dices.append(dice)

    final_miou  = np.mean(ious)
    final_mdice = np.mean(dices)
    print(f"\n{'='*40}")
    print(f"TEST  mIOU: {final_miou:.4f}  |  mDice: {final_mdice:.4f}")
    print(f"{'='*40}")

    with open(os.path.join(SAVE_DIR, "test_results.txt"), "w") as f:
        f.write(f"mIOU:{final_miou:.4f}\nmDice:{final_mdice:.4f}\n")

    # Plots
    ep = range(1, EPOCHS+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history["loss"], 'b-o', linewidth=2, markersize=4)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)

    axes[1].plot(ep, history["miou"], 'g-o', linewidth=2, markersize=4)
    axes[1].axhline(y=final_miou, color='r', linestyle='--', label=f'Test={final_miou:.4f}')
    axes[1].set_title("mIOU"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ep, history["mdice"], 'm-o', linewidth=2, markersize=4)
    axes[2].axhline(y=final_mdice, color='r', linestyle='--', label=f'Test={final_mdice:.4f}')
    axes[2].set_title("mDice"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_plots.png"), dpi=150, bbox_inches='tight')
    print("Plots saved!")


if __name__ == "__main__":
    train()