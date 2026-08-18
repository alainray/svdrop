import numpy as np
import torch
import bisect
import warnings
from torch import randperm, default_generator


class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.group_array = self.get_group_array(re_evaluate=True)
        self.label_array = self.get_label_array(re_evaluate=True)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True):
        """Return an array [g_x1, g_x2, ...]"""
        if re_evaluate:
            group_array = self.dataset.get_group_array()[self.indices]
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenate datasets, preserving group/label arrays.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_group_array(self):
        group_array = []
        for dataset in self.datasets:
            group_array += list(np.squeeze(dataset.get_group_array()))
        return group_array

    def get_label_array(self):
        label_array = []
        for dataset in self.datasets:
            label_array += list(np.squeeze(dataset.get_label_array()))
        return label_array


def get_fold(
    dataset,
    fold=None,
    cross_validation_ratio=0.2,
    num_valid_per_point=4,
    seed=0,
    shuffle=True,
):
    """Returns (train, valid) splits of the dataset."""
    if fold is not None:
        indices = fold.split("_")[1:]
        sweep_ind = int(indices[0])
        fold_ind = int(indices[1])
        assert sweep_ind is None or sweep_ind < num_valid_per_point
        assert fold_ind is None or fold_ind < int(1 / cross_validation_ratio)

    valid_size = int(np.ceil(len(dataset) * cross_validation_ratio))
    num_valid_sets = int(np.ceil(len(dataset) / valid_size))

    random = np.random.RandomState(seed)

    all_folds = []
    for sweep_counter in range(num_valid_per_point):
        folds = []
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        else:
            print("\n" * 10, "WARNING, NOT SHUFFLING", "\n" * 10)
        for i in range(num_valid_sets):
            train_indices = indices[:i * valid_size] + indices[(i + 1) * valid_size:]
            train_split = Subset(dataset, train_indices)

            valid_indices = indices[i * valid_size:(i + 1) * valid_size]
            valid_split = Subset(dataset, valid_indices)

            if sweep_counter == 0 and i == 0:
                print("train_split", train_split, "valid_split", valid_split)
            folds.append((train_split, valid_split))
        all_folds.append(folds)

    if fold is not None:
        train_data_subset, val_data_subset = all_folds[sweep_ind][fold_ind]

        # ⬇️ Import local para evitar ciclo (data.folds -> data.dro_dataset -> data.folds)
        from data import dro_dataset as _dro

        # Wrap en DRODataset Objects
        train_data = _dro.DRODataset(
            train_data_subset,
            process_item_fn=None,
            n_groups=dataset.n_groups,
            n_classes=dataset.n_classes,
            group_str_fn=dataset.group_str,
        )

        val_data = _dro.DRODataset(
            val_data_subset,
            process_item_fn=None,
            n_groups=dataset.n_groups,
            n_classes=dataset.n_classes,
            group_str_fn=dataset.group_str,
        )

        return train_data, val_data
    else:
        # TODO: change this to return DRODatasets not just list.
        return all_folds
