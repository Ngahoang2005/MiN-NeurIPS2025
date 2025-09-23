import os
import numpy as np
from sympy.integrals.meijerint_doc import category
from torchvision import datasets, transforms
from backbones.pretrained_backbone import get_pretrained_backbone
import time
from PIL import Image
from tqdm import tqdm


class iData(object):
    train_trsf = []
    test_trsf = []
    class_order = None


class iCIFAR224(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.class_order = np.arange(100).tolist()

    def data_initialization(self):
        data_path = r'[DATA PATH]'
        train_dataset = datasets.cifar.CIFAR100(data_path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.category_index = train_dataset.classes


class iImageNet_A(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.class_order = np.arange(200).tolist()

    def data_initialization(self):
        train_dir = r'[DATA PATH]'
        test_dir = r'[DATA PATH]'
        self.category_index, self.train_data = split_img_label(train_dir)
        self.category_index, self.test_data = split_img_label(test_dir)


class iFood101(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.class_order = np.arange(101).tolist()

    def data_initialization(self):
        train_dir = r'[DATA PATH]'
        test_dir = r'[DATA PATH]'
        self.category_index, self.train_data = split_img_label(train_dir)
        self.category_index, self.test_data = split_img_label(test_dir)


class iCUB200(iData):
    def __init__(self, args):
        self.args = args

        self.train_data = None
        self.test_data = None
        self.category_index = None

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std
            ),
        ]
        self.class_order = np.arange(200).tolist()

    def data_initialization(self):
        train_dir = r'[DATA PATH]'
        test_dir = r'[DATA PATH]'
        self.category_index, self.train_data = split_img_label(train_dir)
        self.category_index, self.test_data = split_img_label(test_dir)


class iOmnibenchmark(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.class_order = np.arange(300).tolist()

    def data_initialization(self):
        train_dir = r'[DATA PATH]'
        test_dir = r'[DATA PATH]'

        self.category_index, self.train_data = split_img_label(train_dir)
        self.category_index, self.test_data = split_img_label(test_dir)

class iImageNet_R(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
        ]
        self.class_order = np.arange(200).tolist()

    def data_initialization(self):
        train_dir = r'[DATA PATH]'
        test_dir = r'[DATA PATH]'
        self.category_index, self.train_data = split_img_label(train_dir)
        self.category_index, self.test_data = split_img_label(test_dir)


def split_img_label(root):
    category_index = os.listdir(root)
    cat_list = []
    num = 0
    print("start loading the img index!")
    for cat in category_index:
        sample_list = os.listdir(os.path.join(root, cat))
        sample_list = [(os.path.join(root, cat, sample_list[i]), num) for i in range(len(sample_list))]
        cat_list.append(sample_list)
        num += 1
    return category_index, cat_list
