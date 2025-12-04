import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import tqdm

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# SOME PARTS IMPLEMENTED FROM IRKv2 PAPER - SOME ALGORITHMS AND ALL CODE IS OWN WORK:
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
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    return (
        (2.0 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    ).item() * 100


def iou_metric(pred, label, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item() * 100


def precision_metric(pred, label, smooth=1e-6):
    pred = (pred > 0.5).float()
    true_positive = (pred * label).sum()
    predicted_positive = pred.sum()
    return ((true_positive + smooth) / (predicted_positive + smooth)).item() * 100


def recall_metric(pred, label, smooth=1e-6):
    pred = (pred > 0.5).float()
    true_positive = (pred * label).sum()
    actual_positive = label.sum()
    return ((true_positive + smooth) / (actual_positive + smooth)).item() * 100


# ALGORITHMS AND CODE IS OWN WORK
def apply_simple_tta(model, image, use_rotations=False):
    predictions = []

    # 1. Original image
    with torch.no_grad():
        pred_original, *_ = model(image)
        pred_original = torch.sigmoid(pred_original)
        predictions.append(pred_original)

    # 2. Horizontal flip
    image_hflip = torch.flip(image, dims=[3])  # Flip along width (W dimension)
    with torch.no_grad():
        pred_hflip, *_ = model(image_hflip)
        pred_hflip = torch.sigmoid(pred_hflip)
        pred_hflip = torch.flip(pred_hflip, dims=[3])  # Flip back
        predictions.append(pred_hflip)

    # 3. Vertical flip
    image_vflip = torch.flip(image, dims=[2])  # Flip along height (H dimension)
    with torch.no_grad():
        pred_vflip, *_ = model(image_vflip)
        pred_vflip = torch.sigmoid(pred_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[2])  # Flip back
        predictions.append(pred_vflip)

    if use_rotations:
        # 4. Rotate 90° clockwise
        image_rot90 = torch.rot90(image, k=1, dims=[2, 3])  # k=1 means 90° clockwise
        with torch.no_grad():
            pred_rot90, *_ = model(image_rot90)
            pred_rot90 = torch.sigmoid(pred_rot90)
            pred_rot90 = torch.rot90(pred_rot90, k=-1, dims=[2, 3])  # Rotate back
            predictions.append(pred_rot90)

        # 5. Rotate 180°
        image_rot180 = torch.rot90(image, k=2, dims=[2, 3])  # k=2 means 180°
        with torch.no_grad():
            pred_rot180, *_ = model(image_rot180)
            pred_rot180 = torch.sigmoid(pred_rot180)
            pred_rot180 = torch.rot90(pred_rot180, k=-2, dims=[2, 3])  # Rotate back
            predictions.append(pred_rot180)

        # 6. Rotate 270° clockwise (90° counter-clockwise)
        image_rot270 = torch.rot90(image, k=3, dims=[2, 3])  # k=3 means 270° clockwise
        with torch.no_grad():
            pred_rot270, *_ = model(image_rot270)
            pred_rot270 = torch.sigmoid(pred_rot270)
            pred_rot270 = torch.rot90(pred_rot270, k=-3, dims=[2, 3])  # Rotate back
            predictions.append(pred_rot270)

    # Average all predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)

    return avg_prediction


def test_with_simple_tta(model, dataloader, criterion, device, use_rotations=False):
    model.eval()
    total_loss = 0
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    desc = (
        "Testing with Plain TTA (Flips + Rotations)"
        if use_rotations
        else "Testing with Plain TTA (Flips Only)"
    )

    with torch.no_grad():
        for images, masks, paths in tqdm.tqdm(dataloader, desc=desc):
            images = images.to(device)
            masks = masks.to(device)

            # Apply simple TTA (returns averaged sigmoid predictions)
            avg_pred = apply_simple_tta(model, images, use_rotations=use_rotations)

            # Calculate loss using original prediction
            outputs, *_ = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate metrics for each sample in batch
            for i in range(avg_pred.shape[0]):
                dice = dice_metric(avg_pred[i : i + 1], masks[i : i + 1])
                iou = iou_metric(avg_pred[i : i + 1], masks[i : i + 1])
                precision = precision_metric(avg_pred[i : i + 1], masks[i : i + 1])
                recall = recall_metric(avg_pred[i : i + 1], masks[i : i + 1])

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
    parser = argparse.ArgumentParser(
        description="Simple TTA for NAFormer (IRv2-Net style)"
    )

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
    parser.add_argument(
        "--use_rotations",
        action="store_true",
        help="Add rotation augmentations (90°, 180°, 270°) - slower but may improve results",
    )

    # Model
    parser.add_argument(
        "--num_classes", type=int, default=1, help="Number of output classes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Kvasir-SEG/pretrain-NASegFormer.pth",
        help="Path to the pretrained NAFormer checkpoint",
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
    print(f"Loading NAFormer from: {args.model_path}")
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

    # Run test with simple TTA
    print("Starting evaluation with Simple TTA (IRv2-Net style + Rotations)...")
    if args.use_rotations:
        print("TTA Scheme: Original + H-Flip + V-Flip + Rot90° + Rot180° + Rot270°")
        print("Total augmentations: 6 (6x inference time)")
    else:
        print("TTA Scheme: Original + H-Flip + V-Flip (Averaged)")
        print("Total augmentations: 3 (3x inference time)")

    test_loss, test_dice, test_iou, test_precision, test_recall = test_with_simple_tta(
        model, test_loader, criterion, device, use_rotations=args.use_rotations
    )

    # Print results
    print()
    print("NAFormer + Simple TTA TEST RESULTS")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Dice:      {test_dice:.2f}%")
    print(f"Test IoU:       {test_iou:.2f}%")
    print(f"Test Precision: {test_precision:.2f}%")
    print(f"Test Recall:    {test_recall:.2f}%")
    if args.use_rotations:
        print()
        print("Note: Using geometric augmentations with rotations (6x inference time)")
    else:
        print()
        print("Note: Using simple geometric augmentations (3x inference time)")
    print("      Unlike VP-TTA, no visual prompts, AdaBN, or optimization")
    print("      Add --use_rotations flag to include rotation augmentations")


if __name__ == "__main__":
    main()
