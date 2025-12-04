import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import tqdm

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# CODE IS OUR CONTRIBUTION:
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


def dice_metric(pred, label, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    return (
        (2.0 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    ).item() * 100


def iou_metric(pred, label, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item() * 100


def precision_metric(pred, label, smooth=1e-6):
    """Calculate Precision"""
    pred = (pred > 0.5).float()
    true_positive = (pred * label).sum()
    predicted_positive = pred.sum()
    return ((true_positive + smooth) / (predicted_positive + smooth)).item() * 100


def recall_metric(pred, label, smooth=1e-6):
    """Calculate Recall"""
    pred = (pred > 0.5).float()
    true_positive = (pred * label).sum()
    actual_positive = label.sum()
    return ((true_positive + smooth) / (actual_positive + smooth)).item() * 100


def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for images, masks, paths in tqdm.tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            outputs, *_ = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            pred = torch.sigmoid(outputs)

            # Calculate metrics for each sample in batch
            for i in range(pred.shape[0]):
                dice = dice_metric(pred[i : i + 1], masks[i : i + 1])
                iou = iou_metric(pred[i : i + 1], masks[i : i + 1])
                precision = precision_metric(pred[i : i + 1], masks[i : i + 1])
                recall = recall_metric(pred[i : i + 1], masks[i : i + 1])

                all_dice.append(dice)
                all_iou.append(iou)
                all_precision.append(precision)
                all_recall.append(recall)

    avg_loss = total_loss / len(dataloader)
    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    return avg_loss, avg_dice, avg_iou, avg_precision, avg_recall


def main():
    parser = argparse.ArgumentParser(description="Test NA-SegFormer on Test Dataset")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="Kvasir-SEG", help="Dataset name"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data",
        help="Root directory of the dataset",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Testing
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for testing"
    )

    # Model
    parser.add_argument(
        "--num_classes", type=int, default=1, help="Number of output classes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Kvasir-SEG/pretrain-NASegFormer.pth",
        help="Path to the pretrained model checkpoint",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for testing"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dataset
    print()
    print(f"Loading dataset from: {os.path.join(args.dataset_root, args.dataset)}")
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
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Setup model
    print()
    print(f"Loading model from: {args.model_path}")
    model = TSFormer(num_classes=args.num_classes, img_size=args.image_size).to(device)

    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation Dice: {checkpoint.get('best_dice', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)

    # Setup loss
    criterion = BCEDiceLoss()

    # Run test
    print("Starting evaluation on test dataset...")

    test_loss, test_dice, test_iou, test_precision, test_recall = test(
        model, test_loader, criterion, device
    )

    # Print results
    print("TEST RESULTS")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Dice:      {test_dice:.2f}%")
    print(f"Test IoU:       {test_iou:.2f}%")
    print(f"Test Precision: {test_precision:.2f}%")
    print(f"Test Recall:    {test_recall:.2f}%")


if __name__ == "__main__":
    main()
