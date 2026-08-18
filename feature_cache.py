"""Precompute backbone features once so last-layer retraining is cheap.

When --finetune freezes the backbone and --train_bn is off, the backbone is a
fixed function: parameters do not move and the normalization layers run in eval
mode, so they no longer adapt to the batch composition. With --augment_data off
the transforms are deterministic too (Resize + CenterCrop), which makes the
features of every image constant across epochs.

Under those conditions running the full network every epoch is pure waste: a
301-epoch last-layer run spends hours recomputing the same ~12k feature vectors.
This module extracts them once and hands back tensor-backed datasets plus the
head alone, so the training loop in train.py stays exactly the same but each
epoch is a handful of matrix products.

This makes long-budget sweeps (3000+ epochs over a grid of weight decays)
practical, which is the only way to tell a truncated optimisation budget apart
from a regularisation effect.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import dro_dataset


class TensorGroupDataset(Dataset):
    """In-memory (x, y, g, idx) dataset with the interface DRODataset expects."""

    def __init__(self, features, y, g):
        self.features = features
        self.y = y
        self.g = g

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.y[idx], self.g[idx], idx

    def get_group_array(self):
        return self.g.numpy()

    def get_label_array(self):
        return self.y.numpy()


def _split_head(model):
    """Detach the trainable head from the frozen feature extractor."""
    if not hasattr(model, "fc"):
        raise ValueError(
            "--cache_features only supports architectures with a .fc head "
            "(ResNet family). BERT keeps its own classifier."
        )
    head = model.fc
    model.fc = nn.Identity()
    return head


@torch.no_grad()
def _extract(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feats, ys, gs = [], [], []
    for batch in loader:
        x, y, g = batch[0].to(device), batch[1], batch[2]
        feats.append(model(x).detach().cpu())
        ys.append(y)
        gs.append(g)
    return torch.cat(feats), torch.cat(ys), torch.cat(gs)


def build_cached_data(model, data, args, logger, device):
    """Replace the image datasets with cached features and return the head.

    Returns (cached_data, head). `cached_data` has the same keys as `data`, so
    train() does not need to know the difference.
    """
    if args.augment_data:
        raise ValueError(
            "--cache_features needs deterministic transforms; --augment_data "
            "makes the features of an image change from epoch to epoch."
        )
    if args.train_bn:
        raise ValueError(
            "--cache_features needs a fixed backbone; --train_bn lets the "
            "normalization statistics keep adapting during training."
        )
    if args.unfreeze > 0:
        raise ValueError(
            "--cache_features needs a fixed backbone; --unfreeze makes part of "
            "it trainable."
        )

    head = _split_head(model)
    model.eval()
    model = model.to(device)

    cached = {}
    for split in ("train", "val", "test"):
        source = data[f"{split}_data"]
        if source is None:
            cached[f"{split}_data"] = None
            cached[f"{split}_loader"] = None
            continue
        features, y, g = _extract(model, source, args.batch_size,
                                  args.num_workers, device)
        logger.write(f"Cached {split} features: {tuple(features.shape)}\n")
        cached[f"{split}_data"] = dro_dataset.DRODataset(
            TensorGroupDataset(features, y, g),
            process_item_fn=None,
            n_groups=source.n_groups,
            n_classes=source.n_classes,
            group_str_fn=source.group_str,
        )

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0,
                     "pin_memory": True}
    cached["train_loader"] = dro_dataset.get_loader(
        cached["train_data"], train=True,
        reweight_groups=args.reweight_groups, **loader_kwargs)
    for split in ("val", "test"):
        if cached[f"{split}_data"] is not None:
            cached[f"{split}_loader"] = dro_dataset.get_loader(
                cached[f"{split}_data"], train=False, reweight_groups=None,
                **loader_kwargs)

    # The head is the whole model now, so model_norm_sq and reg_loss in the CSVs
    # report the head's L2 rather than the full network's. That is the quantity
    # the weight decay actually acts on.
    return cached, head.to(device)
