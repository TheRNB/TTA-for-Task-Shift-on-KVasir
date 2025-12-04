import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import tqdm
import copy

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# IMPLEMENTED FROM TESTFIT PAPER - CODE IS OWN WORK:
def softmax_entropy(x):
    """Entropy of softmax distribution from logits.
    For binary segmentation, we treat it as 2-class problem."""
    # Convert logits to 2-channel format [background, foreground]
    if x.shape[1] == 1:
        # Binary case: expand to 2 channels
        x_2ch = torch.cat([torch.zeros_like(x), x], dim=1)
    else:
        x_2ch = x
    return -(x_2ch.softmax(1) * x_2ch.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters."""
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ["weight", "bias"] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def dice_metric(pred, label, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    return (
        (2.0 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    ).item() * 100


def iou_metric(pred, label, smooth=1e-6):
    """Calculate IoU"""
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


class TestFit:
    def __init__(self, model, original_model, optimizer, alpha_search_steps=101):
        super().__init__()
        self.model = model
        self.original_model = original_model  # TODO: Keep original model frozen
        self.optimizer = optimizer
        self.alpha_search_steps = alpha_search_steps
        self.loss_function = nn.BCEWithLogitsLoss(reduction="none")

    def find_optimal_alpha(self, adapted_logits, original_logits):
        """Find optimal alpha by minimizing entropy of ensemble predictions."""
        high_entropy = -10000
        low_entropy = 10000
        high_alpha = 0
        low_alpha = 0

        for alpha_step in range(self.alpha_search_steps):
            alpha = alpha_step / 100.0
            ensemble_logits = (
                alpha * adapted_logits.detach() + (1 - alpha) * original_logits
            )

            # Calculate entropy
            entropy = softmax_entropy(ensemble_logits).mean()
            entropy_value = entropy.item()

            if entropy_value >= high_entropy:
                high_entropy = entropy_value
                high_alpha = alpha
            if entropy_value <= low_entropy:
                low_entropy = entropy_value
                low_alpha = alpha

        return low_alpha, high_alpha

    def create_pseudo_labels(self, ensemble_logits, threshold=0.95):
        """Create pseudo-labels from high-confidence predictions."""
        pseudo_labels = torch.sigmoid(ensemble_logits.clone())

        pseudo_labels[pseudo_labels > threshold] = 1.0
        pseudo_labels[pseudo_labels <= threshold] = 0.0

        return pseudo_labels

    def compute_weights(self, pseudo_labels, adapted_logits):
        # Weight 1: Based on pseudo-label confidence
        weight1 = 2 * torch.abs(0.5 - pseudo_labels)
        weight1 = weight1.detach()

        # Weight 2: Based on model prediction confidence
        adapted_probs = torch.sigmoid(adapted_logits.clone())
        weight2 = 2 * torch.abs(0.5 - adapted_probs)
        weight2 = 1 - weight2  # Inverse weight
        weight2 = weight2.detach()

        return weight1, weight2

    def forward_and_adapt(self, x):
        # Get predictions from adapted model
        self.optimizer.zero_grad()
        adapted_logits, *_ = self.model(x)

        # Get predictions from original model (frozen)
        with torch.no_grad():
            original_logits, *_ = self.original_model(x)
            original_logits = original_logits.detach()

        # Find optimal alpha values
        low_alpha, high_alpha = self.find_optimal_alpha(adapted_logits, original_logits)

        # Create output ensemble with low_alpha (low entropy)
        output_logits = low_alpha * adapted_logits + (1 - low_alpha) * original_logits

        # Create pseudo-labels using high_alpha ensemble
        pseudo_label_logits = (
            high_alpha * adapted_logits.detach() + (1 - high_alpha) * original_logits
        )
        pseudo_labels = self.create_pseudo_labels(pseudo_label_logits)

        # Compute instance weights
        weight1, weight2 = self.compute_weights(pseudo_labels, adapted_logits)

        # Compute weighted loss
        loss = self.loss_function(adapted_logits, pseudo_labels.detach())

        # Apply weights (TestFit paper shows both weighted and unweighted versions)
        # Using unweighted version as it's simpler and often works well
        loss = torch.mean(loss)

        # Backward pass and update
        loss.backward()
        self.optimizer.step()

        return output_logits.detach()


def test_testfit(model, dataloader, device, lr=1e-5):
    # Create original model copy (keep frozen)
    original_model = copy.deepcopy(model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    # Configure adapted model for training
    model.train()

    # Setup optimizer for all parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)

    # Create TestFit adapter
    testfit = TestFit(model, original_model, optimizer)

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    print()
    print("TestFit Configuration:")
    print(f"  Learning rate: {lr}")
    print(f"  Alpha search steps: {testfit.alpha_search_steps}")
    print(f"  Optimizing {len(params)} parameters")

    with torch.set_grad_enabled(True):
        for images, masks, paths in tqdm.tqdm(dataloader, desc="TestFit Testing"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward with TestFit adaptation
            outputs = testfit.forward_and_adapt(images)

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

    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    return avg_dice, avg_iou, avg_precision, avg_recall


def main():
    parser = argparse.ArgumentParser(description="TestFit TTA with NAFormer")

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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing (TestFit typically uses 1)",
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

    # TestFit Hyperparameters
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for adaptation"
    )
    parser.add_argument(
        "--alpha_steps", type=int, default=101, help="Number of steps for alpha search"
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

    # Run TestFit
    print("Starting TestFit TTA evaluation...")

    test_dice, test_iou, test_precision, test_recall = test_testfit(
        model, test_loader, device, lr=args.lr
    )

    # Print results
    print("TESTFIT TTA RESULTS")
    print(f"Test Dice:      {test_dice:.2f}%")
    print(f"Test IoU:       {test_iou:.2f}%")
    print(f"Test Precision: {test_precision:.2f}%")
    print(f"Test Recall:    {test_recall:.2f}%")


if __name__ == "__main__":
    main()
