import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tqdm


from networks.TSFormer import TSFormer
from dataloaders.POLYP_dataloader import Kvasir_dataset


# ADAPTED FROM VPTTA CODE
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

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(
            1, C, 1, 1
        ) + self.bias.view(1, C, 1, 1)
        return new_x


# ADAPTED FROM VPTTA CODE
class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=256):
        super().__init__()
        self.prompt_size = (
            int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        )
        self.padding_size = (image_size - self.prompt_size) // 2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
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

        amp_low_ = amp_src[
            :,
            :,
            self.padding_size : self.padding_size + self.prompt_size,
            self.padding_size : self.padding_size + self.prompt_size,
        ]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_


# ADAPTED FROM VPTTA CODE
class Memory(object):
    """
    Create the empty memory buffer
    """

    def __init__(self, size, dimension):
        self.memory = {}
        self.size = size
        self.dimension = dimension

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        for i, key in enumerate(keys):
            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])

            self.memory.update({key.reshape(self.dimension).tobytes(): (logits[i])})

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        from numpy.linalg import norm

        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32
        ).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (
                norm(self.all_keys, axis=1) * norm(key.T)
            )

            K_neighbour_keys = self.all_keys[
                np.argpartition(similarity_scores, -k)[-k:]
            ]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (
                norm(K_neighbour_keys, axis=1) * norm(key.T)
            )
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples)


# OUR CONTRIBUTIONS FROM HERE ONWARDS:


def convert_to_adabn(model, warm_n=5):
    """Convert all BatchNorm2d layers in model to AdaBN"""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Create AdaBN with same parameters
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
    """Change the new_sample status of all AdaBN layers"""
    for module in model.modules():
        if isinstance(module, AdaBN):
            module.new_sample = new_sample


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


class VPTTA_NAFormer:
    def __init__(self, config):
        # Data Loading
        print(
            f"\nLoading dataset from: {os.path.join(config.dataset_root, config.dataset)}"
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
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=config.num_workers,
        )

        print(f"Test samples: {len(test_dataset)}")

        self.image_size = config.image_size
        self.device = config.device
        self.warm_n = config.warm_n
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters
        self.neighbor = config.neighbor

        # Build model
        self.build_model(config)

        # Memory Bank
        self.memory_bank = Memory(
            size=config.memory_size, dimension=self.prompt.data_prompt.numel()
        )

        # Print configuration
        print("VPTTA-NAFormer Configuration")
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()

    def build_model(self, config):
        """Build NAFormer model with VPTTA components"""
        # Create prompt
        self.prompt = Prompt(
            prompt_alpha=self.prompt_alpha, image_size=self.image_size
        ).to(self.device)

        # Load NAFormer model
        print(f"\nLoading NAFormer from: {config.model_path}")
        self.model = TSFormer(
            num_classes=config.num_classes, img_size=self.image_size
        ).to(self.device)

        checkpoint = torch.load(config.model_path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)

        # Convert BatchNorm to AdaBN
        print("Converting BatchNorm layers to AdaBN...")
        self.model = convert_to_adabn(self.model, warm_n=self.warm_n)

        # Setup optimizer for prompt only
        if config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print(f"Number of prompt parameters: {num_params}")

    def run(self):
        """Run VPTTA on test dataset"""
        all_dice = []
        all_iou = []

        print()
        print("Starting VPTTA evaluation...")

        # Test on target domain
        for batch_idx, (x, y, paths) in enumerate(
            tqdm.tqdm(self.test_loader, desc="VPTTA Testing")
        ):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            self.model.eval()
            self.prompt.train()
            change_bn_status(self.model, new_sample=True)

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(x)
                init_data = self.memory_bank.get_neighbours(
                    keys=low_freq.cpu().numpy(), k=self.neighbor
                )
            else:
                init_data = torch.ones(
                    (1, 3, self.prompt.prompt_size, self.prompt.prompt_size)
                ).data

            self.prompt.update(init_data)

            # Train Prompt for n iters
            for tr_iter in range(self.iters):
                prompt_x, _ = self.prompt(x)
                _ = self.model(prompt_x)

                # Collect BN loss from all AdaBN layers
                times, bn_loss = 0, 0
                for module in self.model.modules():
                    if isinstance(module, AdaBN):
                        bn_loss += module.bn_loss
                        times += 1

                loss = bn_loss / times if times > 0 else bn_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                change_bn_status(self.model, new_sample=False)

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(x)
                pred_logit, *_ = self.model(prompt_x)

            # Update the Memory Bank
            self.memory_bank.push(
                keys=low_freq.cpu().numpy(),
                logits=self.prompt.data_prompt.detach().cpu().numpy(),
            )

            # Calculate metrics
            seg_output = torch.sigmoid(pred_logit)

            # Calculate metrics for each sample in batch
            for i in range(seg_output.shape[0]):
                dice = dice_metric(seg_output[i : i + 1], y[i : i + 1])
                iou = iou_metric(seg_output[i : i + 1], y[i : i + 1])
                all_dice.append(dice)
                all_iou.append(iou)

        # Print final results
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)

        print("VPTTA-NAFormer TEST RESULTS")
        print(f"Test Dice: {avg_dice:.2f}%")
        print(f"Test IoU:  {avg_iou:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="VPTTA with NAFormer")

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

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer type: SGD/Adam"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum for SGD")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for Adam")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay")

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (VPTTA typically uses 1)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of adaptation iterations per sample",
    )

    # VPTTA Hyperparameters
    parser.add_argument(
        "--memory_size", type=int, default=40, help="Size of memory bank"
    )
    parser.add_argument(
        "--neighbor", type=int, default=16, help="Number of neighbors to retrieve"
    )
    parser.add_argument(
        "--prompt_alpha", type=float, default=0.01, help="Prompt size ratio"
    )
    parser.add_argument(
        "--warm_n", type=int, default=5, help="Warm-up parameter for AdaBN"
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    config = parser.parse_args()

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {device}")

    # Run VPTTA
    vptta = VPTTA_NAFormer(config)
    vptta.run()


if __name__ == "__main__":
    main()
