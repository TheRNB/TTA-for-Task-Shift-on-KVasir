import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import argparse
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import tqdm
import pickle

from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# CODE FROM OTHER FILES:
class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False
        self.bn_loss = 0

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (new_mu - cur_mu).abs().mean() + (
            new_var.sqrt() - cur_std
        ).abs().mean()

        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(
            1, C, 1, 1
        ) + self.bias.view(1, C, 1, 1)
        return new_x


class Prompt(nn.Module):
    """Visual Prompt for VP-TTA"""

    def __init__(self, prompt_alpha=0.01, image_size=256):
        super().__init__()
        self.prompt_size = (
            int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        )
        self.padding_size = (image_size - self.prompt_size) // 2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        prompt = nn.functional.pad(
            self.data_prompt,
            [
                self.padding_size,
                imgH - self.padding_size - self.prompt_size,
                self.padding_size,
                imgW - self.padding_size - self.prompt_size,
            ],
            mode="constant",
            value=1.0,
        ).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        # iFFT
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real

        return src_in_trg


def convert_to_adabn(model, warm_n=5):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            ada_bn = AdaBN(module.num_features, warm_n=warm_n)
            ada_bn.weight = module.weight
            ada_bn.bias = module.bias
            ada_bn.running_mean = module.running_mean
            ada_bn.running_var = module.running_var
            ada_bn.eps = module.eps
            ada_bn.momentum = module.momentum
            ada_bn.track_running_stats = module.track_running_stats
            setattr(model, name, ada_bn)
        else:
            convert_to_adabn(module, warm_n)
    return model


def change_bn_status(model, new_sample=True):
    for module in model.modules():
        if isinstance(module, AdaBN):
            module.new_sample = new_sample


def dice_metric(pred, label, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * label).sum()
    return (
        (2.0 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    ).item() * 100


def setup_device_and_seed(device_str="cuda:0", seed=43):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    return device


def load_test_dataset(dataset_root, dataset_name, image_size, seed=43):
    dataset = Kvasir_dataset(
        root=os.path.join(dataset_root, dataset_name), target_size=image_size
    )

    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = val_size
    train_size = len(dataset) - val_size - test_size

    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return test_dataset


# CODE IS OWN WORK
def load_model(model_path, num_classes, image_size, device):
    model = TSFormer(num_classes=num_classes, img_size=image_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def vptta_inference(model, prompt, optimizer, image, device, iters=1):
    model.eval()
    prompt.train()
    change_bn_status(model, new_sample=True)

    # Initialize prompt
    init_data = torch.ones((1, 3, prompt.prompt_size, prompt.prompt_size)).data
    prompt.update(init_data)

    # Adapt prompt
    for _ in range(iters):
        prompt_x = prompt(image)
        _ = model(prompt_x)

        # Collect BN loss
        times, bn_loss = 0, 0
        for module in model.modules():
            if isinstance(module, AdaBN):
                bn_loss += module.bn_loss
                times += 1

        loss = bn_loss / times if times > 0 else bn_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        change_bn_status(model, new_sample=False)

    # Final inference
    model.eval()
    prompt.eval()
    with torch.no_grad():
        prompt_x = prompt(image)
        pred_logit, *_ = model(prompt_x)
        pred = torch.sigmoid(pred_logit)

    return pred


def baseline_inference(model, image, device):
    model.eval()
    with torch.no_grad():
        outputs, *_ = model(image)
        pred = torch.sigmoid(outputs)
    return pred


def simple_tta_inference(model, image, device, use_rotations=False):
    model.eval()
    predictions = []

    # Original
    with torch.no_grad():
        pred, *_ = model(image)
        predictions.append(torch.sigmoid(pred))

    # H-flip
    image_hflip = torch.flip(image, dims=[3])
    with torch.no_grad():
        pred_hflip, *_ = model(image_hflip)
        pred_hflip = torch.flip(torch.sigmoid(pred_hflip), dims=[3])
        predictions.append(pred_hflip)

    # V-flip
    image_vflip = torch.flip(image, dims=[2])
    with torch.no_grad():
        pred_vflip, *_ = model(image_vflip)
        pred_vflip = torch.flip(torch.sigmoid(pred_vflip), dims=[2])
        predictions.append(pred_vflip)

    if use_rotations:
        for k in [1, 2, 3]:
            image_rot = torch.rot90(image, k=k, dims=[2, 3])
            with torch.no_grad():
                pred_rot, *_ = model(image_rot)
                pred_rot = torch.rot90(torch.sigmoid(pred_rot), k=-k, dims=[2, 3])
                predictions.append(pred_rot)

    return torch.mean(torch.stack(predictions), dim=0)


def segmentation_entropy(x):
    p = torch.sigmoid(x)
    eps = 1e-6
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return entropy.mean(dim=(1, 2, 3))


def configure_tent_model(model):
    model.train()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    return model


def tent_inference(model, optimizer, image, device, steps=1):
    model.train()

    for _ in range(steps):
        outputs, *_ = model(image)
        loss = segmentation_entropy(outputs).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final prediction
    with torch.no_grad():
        outputs, *_ = model(image)
        pred = torch.sigmoid(outputs)

    return pred


def softmax_entropy_testfit(x):
    if x.shape[1] == 1:
        x_2ch = torch.cat([torch.zeros_like(x), x], dim=1)
    else:
        x_2ch = x
    return -(x_2ch.softmax(1) * x_2ch.log_softmax(1)).sum(1)


def find_optimal_alpha(adapted_logits, original_logits, alpha_steps=101):
    low_entropy = 10000
    low_alpha = 0

    for alpha_step in range(alpha_steps):
        alpha = alpha_step / 100.0
        ensemble_logits = (
            alpha * adapted_logits.detach() + (1 - alpha) * original_logits
        )
        entropy = softmax_entropy_testfit(ensemble_logits).mean().item()

        if entropy <= low_entropy:
            low_entropy = entropy
            low_alpha = alpha

    return low_alpha


def testfit_inference(model, original_model, optimizer, image, device):
    model.train()

    # Get predictions
    optimizer.zero_grad()
    adapted_logits, *_ = model(image)

    with torch.no_grad():
        original_logits, *_ = original_model(image)

    # Find optimal alpha
    low_alpha = find_optimal_alpha(adapted_logits, original_logits)

    # Create ensemble output
    output_logits = low_alpha * adapted_logits + (1 - low_alpha) * original_logits

    # Create pseudo-labels
    pseudo_labels = torch.sigmoid(output_logits.detach())
    pseudo_labels[pseudo_labels > 0.95] = 1.0
    pseudo_labels[pseudo_labels <= 0.95] = 0.0

    # Compute loss and update
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(adapted_logits, pseudo_labels.detach())

    loss.backward()
    optimizer.step()

    # Return prediction
    return torch.sigmoid(output_logits.detach())


def create_confidence_colormap():
    colors = [(1, 0, 0), (1, 1, 1)]
    return mcolors.LinearSegmentedColormap.from_list("confidence", colors, N=100)


def visualize_sample(
    original, gt_mask, pred_mask, confidence, save_path, method_name, dice_score
):
    """Create 4-column visualization with sophisticated confidence map"""
    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.05)

    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(confidence):
        confidence = confidence.cpu().numpy()

    # Handle dimensions
    if original.shape[0] == 3:
        original = np.transpose(original, (1, 2, 0))
    if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask[0]
    if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]
    if len(confidence.shape) == 3 and confidence.shape[0] == 1:
        confidence = confidence[0]

    while len(gt_mask.shape) > 2:
        gt_mask = gt_mask[0]
    while len(pred_mask.shape) > 2:
        pred_mask = pred_mask[0]
    while len(confidence.shape) > 2:
        confidence = confidence[0]

    assert (
        gt_mask.shape == pred_mask.shape == confidence.shape
    ), f"Shape mismatch: gt_mask={gt_mask.shape}, pred_mask={pred_mask.shape}, confidence={confidence.shape}"

    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_mask, cmap="gray")
    ax2.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_mask, cmap="gray")
    ax3.set_title(
        f"Prediction (Dice: {dice_score:.2f}%)", fontsize=12, fontweight="bold"
    )
    ax3.axis("off")

    # TTA Improvement
    ax4 = fig.add_subplot(gs[0, 3])

    # Create RGB confidence map
    confidence_rgb = np.zeros((*confidence.shape, 3))

    gt_binary = (gt_mask > 0.5).astype(bool)
    pred_binary = (pred_mask > 0.5).astype(bool)

    inside_gt = gt_binary
    if np.any(inside_gt):
        confidence_rgb[inside_gt, 0] = 1.0
        confidence_rgb[inside_gt, 1] = (
            np.maximum(confidence[inside_gt] - 0.5, 0.0) + 0.5
        )
        confidence_rgb[inside_gt, 2] = (
            np.maximum(confidence[inside_gt] - 0.5, 0.0) + 0.5
        )
    false_positive = (~gt_binary) & pred_binary
    if np.any(false_positive):
        confidence_rgb[false_positive, 0] = 0.5 * np.maximum(
            confidence[false_positive] - 0.5, 0.0
        )
        confidence_rgb[false_positive, 1] = 1.0 - (
            0.5 * (np.maximum(confidence[false_positive] - 0.5, 0.0))
        )
        confidence_rgb[false_positive, 2] = 1.0 - (
            0.5 * (np.maximum(confidence[false_positive] - 0.5, 0.0))
        )

    ax4.imshow(confidence_rgb)
    ax4.set_title("Confidence Map", fontsize=12, fontweight="bold")
    ax4.axis("off")

    # Legend
    from matplotlib.patches import Rectangle

    legend_elements = [
        Rectangle(
            (0, 0), 1, 1, fc="white", ec="black", linewidth=1, label="TP: High Conf"
        ),
        Rectangle(
            (0, 0), 1, 1, fc="red", ec="black", linewidth=1, label="TP: Low Conf"
        ),
        Rectangle(
            (0, 0), 1, 1, fc="black", ec="white", linewidth=1, label="FP: High Conf"
        ),
        Rectangle(
            (0, 0), 1, 1, fc="blue", ec="black", linewidth=1, label="FP: Low Conf"
        ),
    ]
    ax4.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
    )

    fig.suptitle(f"{method_name}", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_comparison_grid(all_results, sample_idx, save_path):
    methods = list(all_results.keys())
    num_methods = len(methods)

    fig = plt.figure(figsize=(16, 4 * num_methods))
    gs = GridSpec(num_methods, 4, figure=fig, wspace=0.05, hspace=0.1)

    for i, method_name in enumerate(methods):
        result = all_results[method_name][sample_idx]

        original = result["image"].cpu().numpy()
        gt_mask = result["mask"].cpu().numpy()
        pred_mask = (result["pred"] > 0.5).float().cpu().numpy()
        confidence = result["confidence"].cpu().numpy()

        if original.shape[0] == 3:
            original = np.transpose(original, (1, 2, 0))
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[0]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        if len(confidence.shape) == 3:
            confidence = confidence[0]

        # Previous function
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(original)
        if i == 0:
            ax1.set_title("Original", fontsize=12, fontweight="bold")
        ax1.set_ylabel(method_name, fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(gt_mask, cmap="gray")
        if i == 0:
            ax2.set_title("Ground Truth", fontsize=12, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(pred_mask, cmap="gray")
        if i == 0:
            ax3.set_title("Prediction", fontsize=12, fontweight="bold")
        ax3.text(
            0.5,
            -0.1,
            f"Dice: {result['dice']:.2f}%",
            transform=ax3.transAxes,
            ha="center",
            fontsize=10,
        )
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[i, 3])

        confidence_rgb = np.zeros((*confidence.shape, 3))

        gt_binary = (gt_mask > 0.5).astype(bool)
        pred_binary = (pred_mask > 0.5).astype(bool)

        inside_gt = gt_binary
        if np.any(inside_gt):
            confidence_rgb[inside_gt, 0] = 1.0
            confidence_rgb[inside_gt, 1] = 0.5 * (
                np.maximum(confidence[inside_gt] - 0.5, 0.0)
            )
            confidence_rgb[inside_gt, 2] = 0.5 * (
                np.maximum(confidence[inside_gt] - 0.5, 0.0)
            )
        false_positive = (~gt_binary) & pred_binary
        if np.any(false_positive):
            confidence_rgb[false_positive, 0] = 0.5
            confidence_rgb[false_positive, 1] = 1.0 - 0.5 * (
                np.maximum(confidence[false_positive] - 0.5, 0.0)
            )
            confidence_rgb[false_positive, 2] = 1.0 - 0.5 * (
                np.maximum(confidence[false_positive] - 0.5, 0.0)
            )

        ax4.imshow(confidence_rgb)
        if i == 0:
            ax4.set_title("Confidence", fontsize=12, fontweight="bold")
        ax4.axis("off")

    fig.suptitle(
        f"Comparison of All Methods - Sample {sample_idx}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize All TTA Methods Including VP-TTA"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="Kvasir-SEG")
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=256)

    # Model
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument(
        "--model_path", type=str, default="./models/Kvasir-SEG/pretrain-NASegFormer.pth"
    )

    # Visualization
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--output_dir", type=str, default="./tta_visualizations_complete"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results as pickle for later analysis",
    )
    parser.add_argument(
        "--create_comparison_grids",
        action="store_true",
        help="Create comparison grids showing all methods side-by-side",
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Setup
    device = setup_device_and_seed(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print()
    print("Loading test dataset...")
    test_dataset = load_test_dataset(args.dataset_root, args.dataset, args.image_size)

    num_viz = min(args.num_samples, len(test_dataset))
    viz_dataset = Subset(test_dataset, list(range(num_viz)))
    viz_loader = DataLoader(viz_dataset, batch_size=1, shuffle=False)

    print(f"Visualizing {num_viz} samples")
    print()

    all_results = {}

    # Baseline
    print()
    print("Running Baseline")
    print()
    model = load_model(args.model_path, args.num_classes, args.image_size, device)
    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="Baseline"):
        image, mask = image.to(device), mask.to(device)
        pred = baseline_inference(model, image, device)
        dice = dice_metric(pred, mask)
        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["Baseline"] = results

    # Simple TTA
    print()
    print("Running Simple TTA")
    print()
    model = load_model(args.model_path, args.num_classes, args.image_size, device)
    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="Plain TTA"):
        image, mask = image.to(device), mask.to(device)
        pred = simple_tta_inference(model, image, device)
        dice = dice_metric(pred, mask)
        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["Simple TTA"] = results

    # Simple TTA with Rotations
    print()
    print("Running Simple TTA with Rotations")
    print()
    model = load_model(args.model_path, args.num_classes, args.image_size, device)
    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="Plain TTA + Rotations"):
        image, mask = image.to(device), mask.to(device)
        pred = simple_tta_inference(model, image, device, use_rotations=True)
        dice = dice_metric(pred, mask)
        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["Simple TTA + Rotations"] = results

    # Tent
    print()
    print("Running Tent TTA")
    print()
    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="Tent TTA"):
        # Reload model for each sample (episodic)
        model_tent = load_model(
            args.model_path, args.num_classes, args.image_size, device
        )
        model_tent = configure_tent_model(model_tent)

        params = [
            p
            for nm, m in model_tent.named_modules()
            for np, p in m.named_parameters()
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))
            and np in ["weight", "bias"]
        ]

        optimizer_tent = torch.optim.SGD(params, lr=0.00025, momentum=0.9)

        image, mask = image.to(device), mask.to(device)
        pred = tent_inference(model_tent, optimizer_tent, image, device, steps=1)
        dice = dice_metric(pred, mask)

        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["Tent"] = results

    # TestFit
    print()
    print("Running TestFit TTA")
    print()
    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="TestFit TTA"):
        # Reload model for each sample
        model_testfit = load_model(
            args.model_path, args.num_classes, args.image_size, device
        )
        model_testfit.train()

        model_original = deepcopy(model_testfit)
        model_original.eval()
        for param in model_original.parameters():
            param.requires_grad = False

        params = [p for p in model_testfit.parameters() if p.requires_grad]
        optimizer_testfit = torch.optim.SGD(params, lr=1e-5)

        image, mask = image.to(device), mask.to(device)
        pred = testfit_inference(
            model_testfit, model_original, optimizer_testfit, image, device
        )
        dice = dice_metric(pred, mask)

        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["TestFit"] = results

    # VP-TTA
    print()
    print("Running VP-TTA")
    print()
    model = load_model(args.model_path, args.num_classes, args.image_size, device)
    model = convert_to_adabn(model, warm_n=5)

    results = []

    for image, mask, _ in tqdm.tqdm(viz_loader, desc="VP-TTA"):
        # Create fresh prompt for each sample
        prompt = Prompt(prompt_alpha=0.01, image_size=args.image_size).to(device)
        optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)

        image, mask = image.to(device), mask.to(device)
        pred = vptta_inference(model, prompt, optimizer, image, device, iters=1)
        dice = dice_metric(pred, mask)

        results.append(
            {
                "image": image[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu(),
                "confidence": pred[0].cpu(),
                "dice": dice,
            }
        )
    all_results["VP-TTA"] = results

    # Generate Visualizations
    print()
    print("Generating Visualizations")
    print()

    for method_name, results in all_results.items():
        method_dir = os.path.join(args.output_dir, method_name.replace(" ", "_"))
        os.makedirs(method_dir, exist_ok=True)

        print()
        print(f"Saving {method_name} visualizations")
        for idx, result in enumerate(tqdm.tqdm(results)):
            pred_binary = (result["pred"] > 0.5).float()
            save_path = os.path.join(method_dir, f"sample_{idx:03d}.pdf")

            visualize_sample(
                result["image"],
                result["mask"],
                pred_binary,
                result["confidence"],
                save_path,
                method_name,
                result["dice"],
            )

    # Create comparison grids
    if args.create_comparison_grids:
        print()
        print("Creating Comparison Grids")
        print()

        comp_dir = os.path.join(args.output_dir, "Comparisons")
        os.makedirs(comp_dir, exist_ok=True)

        for idx in tqdm.tqdm(range(num_viz), desc="Generating grids"):
            save_path = os.path.join(comp_dir, f"comparison_sample_{idx:03d}.pdf")
            create_comparison_grid(all_results, idx, save_path)

    # Save results
    if args.save_results:
        results_path = os.path.join(args.output_dir, "all_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        print()
        print(f"Results saved to: {results_path}")

    # Summary
    print()
    print("Summary Statistics")
    print()

    for method_name, results in all_results.items():
        dice_scores = [r["dice"] for r in results]
        print(
            f"{method_name:15s}: Dice = {np.mean(dice_scores):.2f}% ± {np.std(dice_scores):.2f}%"
        )

    print()
    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
