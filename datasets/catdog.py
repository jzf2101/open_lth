# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision
from torch.utils.data import Subset

from datasets import base
from platforms.platform import get_platform


class CatDog(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, "w") as fp:
                sys.stdout = fp
                super(CatDog, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples():
        return 50000 * 0.2

    @staticmethod
    def num_test_examples():
        return 10000 * 0.2

    @staticmethod
    def num_classes():
        return 10 * 0.2

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),
        ]
        train_set = CatDog(
            train=True,
            root=os.path.join(get_platform().dataset_root, "cifar10"),
            download=True,
        )
        train_set = Dataset(
            train_set.data,
            np.array(train_set.targets),
            augment if use_augmentation else [],
        )
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        train_idx = [
            i
            for i, f in enumerate(train_set._labels)
            if f in [classes.index("cat"), classes.index("dog")]
        ]

        return Subset(train_set, train_idx)

    @staticmethod
    def get_test_set():
        test_set = CatDog(
            train=False,
            root=os.path.join(get_platform().dataset_root, "cifar10"),
            download=True,
        )
        test_set = Dataset(test_set.data, np.array(test_set.targets))

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        test_idx = [
            i
            for i, f in enumerate(test_set._labels)
            if f in [classes.index("cat"), classes.index("dog")]
        ]

        return Subset(test_set, test_idx)

    def __init__(self, examples, labels, image_transforms=None):
        super(Dataset, self).__init__(
            examples,
            labels,
            image_transforms or [],
            [
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ],
        )

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
