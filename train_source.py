import os
from random import seed
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# IMPLEMENTED FROM NA-SEGFORMER PAPER - CODE IS OWN WORK
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


# IMPLEMENTED FROM NA-SEGFORMER PAPER - CODE IS OWN WORK
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# OUR CONTRIBUTIONS FROM HERE ONWARDS:
def dice_metric(pred, label, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    return (
        (2.0 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    ).item() * 100


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_dice = []

    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs, *_ = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            pred = torch.sigmoid(outputs)
            dice = dice_metric(pred, masks)
            all_dice.append(dice)

    return total_loss / len(dataloader), np.mean(all_dice)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs, *_ = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            pred = torch.sigmoid(outputs)
            dice = dice_metric(pred, masks)
            all_dice.append(dice)

    return total_loss / len(dataloader), np.mean(all_dice)


def main():
    parser = argparse.ArgumentParser(description="Train NA-SegFormer on Source Domain")
    parser.add_argument("--dataset", type=str, default="Kvasir-SEG")
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Model
    parser.add_argument("--num_classes", type=int, default=1)

    # Paths
    parser.add_argument("--save_dir", type=str, default="./models")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Setup directories
    save_path = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_path, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dataset
    dataset = Kvasir_dataset(
        root=os.path.join(args.dataset_root, args.dataset), target_size=args.image_size
    )

    # Split dataset (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = val_size
    train_size = len(dataset) - val_size - test_size

    torch.manual_seed(43)
    np.random.seed(43)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"test samples: {len(test_dataset)}")

    # Setup model
    model = TSFormer(num_classes=args.num_classes, img_size=args.image_size).to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Setup loss
    criterion = BCEDiceLoss()

    # Training loop
    best_dice = 0

    print()
    print("Starting training...")

    with tqdm.tqdm(total=args.epochs, desc="Training") as pbar:
        for epoch in range(args.epochs):
            train_loss, train_dice = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_dice = validate(model, val_loader, criterion, device)

            scheduler.step()

            pbar.update(1)

            pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Train Dice": f"{train_dice:.2f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Val Dice": f"{val_dice:.2f}%",
                }
            )

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_dice": best_dice,
                    },
                    os.path.join(save_path, "pretrain-NASegFormer.pth"),
                )
                pbar.write(f"  -> Saved best model (Dice: {best_dice:.2f}%)")

    print(f"Training complete! Best Val Dice: {best_dice:.2f}%")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
