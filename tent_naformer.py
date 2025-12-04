import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from copy import deepcopy
from torch.utils.data import DataLoader
import tqdm

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# ADAPTED FROM TENT CODE
class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        # Ensure gradients are enabled for adaptation
        with torch.enable_grad():
            # Perform adaptation steps
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


# ADAPTED FROM TENT CODE BUT CHANGED BY US FOR OUR TASK
def segmentation_entropy(x: torch.Tensor) -> torch.Tensor:
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(x)

    # For binary segmentation, compute entropy: -p*log(p) - (1-p)*log(1-p)
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

    # Average over spatial dimensions (H, W) and channel dimension
    # Shape: [B, C, H, W] -> [B]
    return entropy.mean(dim=(1, 2, 3))


# ADAPTED FROM TENT CODE
def forward_and_adapt(x, model, optimizer):
    # Forward pass with gradient computation
    outputs, *_ = model(x)  # NAFormer returns (main_output, aux1, aux2, aux3)

    # Compute entropy loss
    loss = segmentation_entropy(outputs).mean(0)

    # Check if loss has gradients
    if not loss.requires_grad:
        raise RuntimeError(
            f"Loss does not require gradients!\n"
            f"  loss.requires_grad: {loss.requires_grad}\n"
            f"  outputs.requires_grad: {outputs.requires_grad}\n"
            f"  model.training: {model.training}\n"
            "Check that BatchNorm parameters are correctly configured."
        )

    # Backward pass and update
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


# ADAPTED FROM TENT CODE
def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


# ADAPTED FROM TENT CODE
def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


# ADAPTED FROM TENT CODE
def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


# ADAPTED FROM TENT CODE BUT CHANGED BY US FOR OUR TASK
def configure_model(model):
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    bn_count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            bn_count += 1
        elif isinstance(m, nn.LayerNorm):
            # Also enable LayerNorm parameters for adaptation
            m.requires_grad_(True)
            bn_count += 1

    if bn_count == 0:
        raise ValueError(
            "No BatchNorm2d or LayerNorm layers found in model! Tent requires normalization layers."
        )

    print(f"Configured {bn_count} normalization layers for adaptation")
    return model


# ADAPTED FROM TENT CODE BUT CHANGED BY US FOR OUR TASK
def check_model(model):
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " "check which require grad"
    assert not has_all_params, (
        "tent should not update all params: " "check which require grad"
    )
    has_norm = any(
        [isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)) for m in model.modules()]
    )
    assert (
        has_norm
    ), "tent needs normalization (BatchNorm2d or LayerNorm) for its optimization"


# REST OF OUR CONTRIBUTIONS FROM HERE ONWARDS:
def dice_metric(pred, label, smooth=1e-6):
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


class Tent_NAFormer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.episodic = config.episodic
        self.steps = config.steps

        # Data Loading
        print()
        print(
            f"Loading dataset from: {os.path.join(config.dataset_root, config.dataset)}"
        )
        dataset = Kvasir_dataset(
            root=os.path.join(config.dataset_root, config.dataset),
            target_size=config.image_size,
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

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        print(f"Test samples: {len(test_dataset)}")

        # Build model with Tent
        self.build_tent_model(config)

        # Print configuration
        print()
        print("Tent-NAFormer Configuration")
        print(f"Episodic mode: {self.episodic}")
        print(f"Adaptation steps: {self.steps}")
        print(f"Learning rate: {config.lr}")
        print(f"Batch size: {config.batch_size}")
        self.print_bn_params()

    def build_tent_model(self, config):
        """Build NAFormer model with Tent adaptation"""
        # Load NAFormer model
        print()
        print(f"Loading NAFormer from: {config.model_path}")
        self.base_model = TSFormer(
            num_classes=config.num_classes, img_size=config.image_size
        ).to(self.device)

        checkpoint = torch.load(config.model_path, map_location=self.device)
        if "model" in checkpoint:
            self.base_model.load_state_dict(checkpoint["model"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(
                f"Best validation Dice: {checkpoint.get('best_dice', 'unknown'):.2f}%"
            )
        else:
            self.base_model.load_state_dict(checkpoint)

        # Configure model for Tent
        print("Configuring model for Tent adaptation...")
        self.base_model = configure_model(self.base_model)

        # Verify configuration
        print(f"Model training mode: {self.base_model.training}")

        # Collect only batch norm parameters
        params, param_names = collect_params(self.base_model)
        print(f"Number of adaptable BN parameters: {len(params)}")

        if len(params) == 0:
            raise ValueError(
                "No BatchNorm parameters found! Tent requires BatchNorm layers."
            )

        # Verify that parameters actually require grad
        params_with_grad = sum(1 for p in params if p.requires_grad)
        print(
            f"BN parameters with requires_grad=True: {params_with_grad}/{len(params)}"
        )

        if params_with_grad == 0:
            raise ValueError("No BN parameters have requires_grad=True!")

        # Setup optimizer for BN parameters only
        self.optimizer = torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            dampening=config.dampening,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )

        # Test gradient flow with a dummy forward pass
        print("Testing gradient flow...")
        dummy_input = torch.randn(1, 3, config.image_size, config.image_size).to(
            self.device
        )
        with torch.enable_grad():
            test_output, *_ = self.base_model(dummy_input)
            if test_output.requires_grad:
                print("Gradient flow test passed: outputs require grad")
            else:
                print("WARNING: Gradient flow test failed: outputs do NOT require grad")
                # Try to diagnose the issue
                for name, module in self.base_model.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        print(f"  Norm layer: {name} (type: {type(module).__name__})")
                        print(
                            f"    weight.requires_grad: {module.weight.requires_grad if module.weight is not None else 'N/A'}"
                        )
                        print(
                            f"    bias.requires_grad: {module.bias.requires_grad if module.bias is not None else 'N/A'}"
                        )
                        break  # Just show first one

        # Wrap model with Tent
        self.model = Tent(
            self.base_model, self.optimizer, steps=self.steps, episodic=self.episodic
        )

        # Check model compatibility
        try:
            check_model(self.base_model)
            print("Model check passed: Compatible with Tent")
        except AssertionError as e:
            print(f"Warning: Model check failed: {e}")
            print("  Attempting to continue anyway...")

    def print_bn_params(self):
        """Print number of normalization parameters being adapted"""
        num_params = 0
        num_norm_layers = 0
        for m in self.base_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                num_norm_layers += 1
                for p in m.parameters():
                    if p.requires_grad:
                        num_params += p.numel()
        print(f"Number of normalization layers: {num_norm_layers}")
        print(f"Number of adaptable parameters: {num_params}")

    def run(self):
        """Run Tent TTA on test dataset"""
        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []

        print()
        print("Starting Tent TTA evaluation...")

        # Test with Tent adaptation
        for images, masks, paths in tqdm.tqdm(
            self.test_loader, desc="Tent TTA Testing"
        ):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Tent forward (includes adaptation with gradients enabled internally)
            outputs = self.model(images)

            # Get predictions (detach to avoid keeping computation graph)
            with torch.no_grad():
                pred = torch.sigmoid(outputs.detach())

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

        # Print final results
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)

        print()
        print("Tent-NAFormer TEST RESULTS")
        print(f"Test Dice:      {avg_dice:.2f}%")
        print(f"Test IoU:       {avg_iou:.2f}%")
        print(f"Test Precision: {avg_precision:.2f}%")
        print(f"Test Recall:    {avg_recall:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Tent TTA with NAFormer")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="Kvasir-SEG", help="Dataset name"
    )
    parser.add_argument(
        "--dataset_root", type=str, default="./data", help="Root directory of datasets"
    )
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model
    parser.add_argument(
        "--num_classes", type=int, default=1, help="Number of output classes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Kvasir-SEG/pretrain-NASegFormer.pth",
        help="Path to pretrained NAFormer model",
    )

    # Tent Hyperparameters
    parser.add_argument(
        "--episodic",
        action="store_true",
        default=False,
        help="Reset model after each sample (episodic mode)",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of adaptation steps per sample"
    )

    # Optimizer
    parser.add_argument(
        "--lr", type=float, default=0.00025, help="Learning rate for BN adaptation"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument(
        "--dampening", type=float, default=0.0, help="Dampening for momentum"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--nesterov", action="store_true", default=False, help="Use Nesterov momentum"
    )

    # Testing
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (Use 1 for TTA)"
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    config = parser.parse_args()

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {device}")

    # Run Tent TTA
    tent = Tent_NAFormer(config)
    tent.run()


if __name__ == "__main__":
    main()
