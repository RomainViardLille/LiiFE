import os
from typing import List, Dict
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
import glob

# monai_unet_pipeline.py
# Exemple minimal d'utilisation de MONAI pour entraîner un UNet sur
# deux listes de fichiers NIfTI : images_list et labels_list.
# Ajuster paramètres (augmentation, out_channels) selon vos labels.



from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)


def make_datalist(images: List[str], labels: List[str]) -> List[Dict[str, str]]:
    assert len(images) == len(labels), "lists must have same length"
    return [{"image": i, "label": l} for i, l in zip(images, labels)]


def get_transforms(spacing=(1.0, 1.0, 1.0), intensity_window=(0, 1500)):
    # Pipeline de transformations (load, reorienter, resample, normaliser, augmentations)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            #AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=intensity_window[0], a_max=intensity_window[1], b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=intensity_window[0], a_max=intensity_window[1], b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def get_dataloaders(
    images: List[str],
    labels: List[str],
    batch_size: int = 1,
    val_split: float = 0.2,
    cache_rate: float = 1.0,
    num_workers: int = 4,
):
    datalist = make_datalist(images, labels)
    n_val = int(len(datalist) * val_split)
    train_list = datalist[n_val:]
    val_list = datalist[:n_val] if n_val > 0 else []

    train_transforms, val_transforms = get_transforms()

    train_ds = CacheDataset(data=train_list, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_list:
        val_ds = CacheDataset(data=val_list, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def create_unet(in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)):
    # UNet standard 3D
    net = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        norm="INSTANCE",
        deep_supervision=False,
    )
    return net


def train(
    images: List[str],
    labels: List[str],
    out_channels: int = 2,
    max_epochs: int = 50,
    lr: float = 1e-4,
    device: str = None,
):
    set_determinism(seed=42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(images, labels, batch_size=1)

    model = create_unet(in_channels=1, out_channels=out_channels).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True) if out_channels > 1 else DiceLoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)
            optimizer.zero_grad()
            outputs = model(img)
            # outputs shape: (B, C, D, H, W)
            loss = loss_fn(outputs, lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch}/{max_epochs} - train loss: {epoch_loss:.4f}")

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                for val_batch in val_loader:
                    val_img = val_batch["image"].to(device)
                    val_lbl = val_batch["label"].long().to(device)
                    val_outputs = model(val_img)
                    if out_channels > 1:
                        # convert to one-hot for metric
                        val_outputs = torch.softmax(val_outputs, dim=1)
                    else:
                        val_outputs = torch.sigmoid(val_outputs)
                    dice_metric(y_pred=val_outputs, y=val_lbl)
                metric = dice_metric.aggregate().item()
                print(f"Validation Dice: {metric:.4f}")

    return model

