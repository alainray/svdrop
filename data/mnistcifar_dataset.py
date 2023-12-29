import torch
from torch.utils.data import TensorDataset
from easydict import EasyDict as edict
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from data.confounder_dataset import ConfounderDataset
import os
import pandas as pd
from models import model_attributes
import numpy as np
from data.folds import Subset
class MNISTCIFARDataset(ConfounderDataset):
    """
    MNISTCIFAR dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names, # confounder_names should be in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        augment_data=False,
        model_type=None,
        metadata_csv_name="metadata.csv"
    ):
        self.root_dir = root_dir
        self.target_name = "CIFAR"
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir,
            f"MNISTCIFAR/MNIST_CIFAR_binary_{confounder_names[0]}.pth") 
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )
        self.confounder_names = ["MNIST"]
        # Read in metadata
        #print(f"Reading '{os.path.join(self.data_dir, metadata_csv_name)}'")
        #self.metadata_df = pd.read_csv(
        #    os.path.join(self.data_dir, metadata_csv_name))
        self.data = torch.load(self.data_dir) # load full tensor dataset
        # Get the y values
        j = 0
        self.y_array  = []
        self.group_array = []
        self.index_array = []
        self.data_array = []
        self.split_array = []
        self.confounder_array = []
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        # Populate data from Tensor Dataset
        for split in ['train','val','test']:
            for x, y, c in zip(self.data[split]['data'],self.data[split]['targets'],self.data[split]['mnist']):
                self.index_array.append(j)
                self.data_array.append(x)
                self.y_array.append(y)
                self.confounder_array.append(c)
                self.split_array.append(self.split_dict[split])
                j+=1
        self.n_classes = 2
        # Convert to numpy
        self.data_array = torch.from_numpy(np.array(self.data_array))
        self.y_array = np.array(self.y_array)
        #self.y_array = (self.y_array > 4).astype(int)                   # binarize labels
        self.index_array = np.array(self.index_array)
        self.confounder_array = np.array(self.confounder_array)
        #self.confounder_array = (self.confounder_array > 4).astype(int) # binarize labels
        self.split_array = np.array(self.split_array)
        # We only support one confounder for MNISTCIFAR: MNIST
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        assert self.n_groups == 4, "check the code if you are running otherwise"
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype("int")

        # Extract filenames and splits
        # self.filename_array = self.metadata_df["img_filename"].values


        # Set transform
        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            self.features_mat = torch.from_numpy(
                np.load(
                    os.path.join(
                        root_dir,
                        "features",
                        model_attributes[self.model_type]["feature_filename"],
                    ))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_mnistcifar(self.model_type,
                                                     train=True,
                                                     augment_data=augment_data)
            self.eval_transform = get_transform_mnistcifar(self.model_type,
                                                    train=False,
                                                    augment_data=augment_data)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
       

        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            x = self.features_mat[idx, :]
        else:
            x = self.data_array[idx]
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict[
                    "train"] and self.train_transform:
                x = self.train_transform(x)
            elif (self.split_array[idx]
                  in [self.split_dict["val"], self.split_dict["test"]]
                  and self.eval_transform):
                x = self.eval_transform(x)
            # Flatten if needed
            if model_attributes[self.model_type]["flatten"]:
                assert x.dim() == 3
                x = x.view(-1)

        return x, y, g, idx
    
    def __len__(self):
        return len(self.index_array)
    
    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val",
                             "test"), f"{split} is not a valid split"
            mask = self.split_array == self.split_dict[split]

            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(
                    np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets
    
def get_transform_mnistcifar(model_type, train, augment_data):
    n_channels = 3
    tf = [
    tt.Normalize([0.5] * n_channels, [0.5] * n_channels)]

    return tt.Compose(tf)


def mnist_cifar(root, split, binarize=False):

    all_splits = torch.load(root)
    sp = split
    if split == "id":
        sp = 'train'
    ds = all_splits[sp]
    if binarize:
        ds['targets'] = (ds['targets'] > 4).float()

    dataset = TensorDataset(ds['data'], ds['targets'], ds['group'])

    if split in ['train', 'id']:
        generator1 = torch.Generator().manual_seed(42)
        dss = random_split(dataset, [9000, 1000], generator=generator1 )
        if split == 'train':
            return dss[0]
        elif split == 'id':
            return dss[1]

    return dataset


if __name__ == '__main__':
    ds = MNISTCIFARDataset(
    "../datasets",
    "mnisticifar",
    "0.9", # confounder_names should be in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    augment_data=False,
    model_type="resnet50")
    print(ds[0])
   