import os
import random
from pathlib import Path

from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from PIL import Image


class Isic2019Raw(torch.utils.data.Dataset):
    """Pytorch dataset containing all the features, labels and datacenter
    information for Isic2019.

    Attributes
    ----------
    image_paths: list[str]
        the list with the path towards all features
    targets: list[int]
        the list with all classification labels for all features
    centers: list[int]
        the list for all datacenters for all features
    X_dtype: torch.dtype
        the dtype of the X features output
    y_dtype: torch.dtype
        the dtype of the y label output
    augmentations:
        image transform operations from the albumentations library,
        used for data augmentation
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Parameters
    ----------
    X_dtype :
    y_dtype :
    augmentations :
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
        data_path=None,
    ):
        """
        Cf class docstring
        """
        input_path = data_path

        dir = str(Path(os.path.realpath(__file__)).parent.resolve())
        self.dic = {
            "input_preprocessed": os.path.join(
                input_path, "ISIC_2019_Training_Input"
            ),
            "train_test_split": os.path.join(
                input_path, "train_test_split.txt"
            ),
        }
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        df2 = pd.read_csv(self.dic["train_test_split"])
        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target
        self.augmentations = augmentations
        self.centers = df2.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        target = self.targets[idx]

        # Image augmentations
        if self.augmentations is not None:
            image = self.augmentations(image)
        image = image.permute(2, 0, 1).to(self.X_dtype)
        return (
            image,
            torch.tensor(target, dtype=self.y_dtype),
            image_path
        )


class FedIsic2019(Isic2019Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for the Isic2019 federated classification.
    One can instantiate this dataset with train or test data coming from either of
    the 6 centers it was created from or all data pooled.
    The train/test split is fixed and given in the train_test_split file.

    Parameters
    ----------
    center : int, optional
        Default to 0
    train : bool, optional
        Default to True
    pooled : bool, optional
        Default to False
    debug : bool, optional
        Default to False
    X_dtype : torch.dtype, optional
        Default to torch.float32
    y_dtype : torch.dtype, optional
        Default to torch.int64
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        debug: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        data_path: str = None,
    ):
        """Cf class docstring"""
        sz = 200
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        if train:
            augmentations = transforms.Compose([transforms.ToTensor(),
                transforms.RandomAffine(degrees=(-50, 50), scale=(0.93, 1.07)), 
                transforms.ColorJitter(brightness=0.15, contrast=0.1),
                transforms.RandomHorizontalFlip(p=0.5),  
                transforms.RandomAffine(degrees=0, shear=0.1),  
                transforms.RandomCrop(sz),  
                transforms.RandomErasing(p=0.5, scale=(1/16, 1/8), ratio=(1, 1)), 
                transforms.Normalize(mean, std), 
            ])
        else:
            augmentations = transforms.Compose([transforms.ToTensor(),
                transforms.CenterCrop(sz),
                transforms.Normalize(mean, std),  
            ])

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations= augmentations,
            data_path=data_path,
        )

        self.center = center
        self.train_test = "train" if train else "test"
        self.pooled = pooled
        self.key = self.train_test + "_" + str(self.center)
        df = pd.read_csv(self.dic["train_test_split"])

        if self.pooled:
            df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)

        if not self.pooled:
            assert center in range(6)
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target
        self.centers = df2.center


if __name__ == "__main__":

    mydataset = Isic2019Raw()
    print("Example of dataset record: ", mydataset[0])
    print(f"The dataset has {len(mydataset)} elements")
    for i in range(10):
        print(f"Size of image {i} ", mydataset[i][0].shape)
        print(f"Target {i} ", mydataset[i][1])

    mydataset = FedIsic2019(train=True, pooled=True)
    print(len(mydataset))
    print("Size of image 0 ", mydataset[0][0].shape)
    mydataset = FedIsic2019(train=False, pooled=True)
    print(len(mydataset))
    print("Size of image 0 ", mydataset[0][0].shape)

    for i in range(6):
        mydataset = FedIsic2019(center=i, train=True, pooled=False)
        print(len(mydataset))
        print("Size of image 0 ", mydataset[0][0].shape)
        mydataset = FedIsic2019(center=i, train=False, pooled=False)
        print(len(mydataset))
        print("Size of image 0 ", mydataset[0][0].shape)

    mydataset = FedIsic2019(center=5, train=False, pooled=False)
    print(len(mydataset))
    for i in range(11):
        print(f"Size of image {i} ", mydataset[i][0].shape)
