from torch.utils import data
from PIL import Image
import os
import torchvision.transforms as transforms


# CODE IS OWN WORK
class POLYP_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [transforms.Resize(self.target_size), transforms.ToTensor()]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file).convert("RGB")
        label = Image.open(label_file).convert("L")

        img = self.img_transform(img)
        label = self.gt_transform(label)

        return img, label, img_file


class Kvasir_dataset(data.Dataset):
    def __init__(self, root, target_size=352):
        super().__init__()
        self.root = root
        self.target_size = (target_size, target_size)

        # Find images and masks directories
        images_dir = os.path.join(root, "images")
        masks_dir = os.path.join(root, "masks")

        if not os.path.exists(images_dir):
            images_dir = os.path.join(root, "Images")
        if not os.path.exists(masks_dir):
            masks_dir = os.path.join(root, "Masks")
            if not os.path.exists(masks_dir):
                masks_dir = os.path.join(root, "GT")

        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # Get all image files
        self.img_files = sorted(
            [
                f
                for f in os.listdir(images_dir)
                if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
        self.len = len(self.img_files)

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [transforms.Resize(self.target_size), transforms.ToTensor()]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_name = self.img_files[item]
        img_file = os.path.join(self.images_dir, img_name)

        # Find corresponding mask
        base_name = os.path.splitext(img_name)[0]
        mask_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(self.masks_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_file = candidate
                break

        if mask_file is None:
            mask_file = os.path.join(self.masks_dir, img_name)

        img = Image.open(img_file).convert("RGB")
        label = Image.open(mask_file).convert("L")

        img = self.img_transform(img)
        label = self.gt_transform(label)

        return img, label, img_file
