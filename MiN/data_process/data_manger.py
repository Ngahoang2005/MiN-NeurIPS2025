import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from data_process.data import iCIFAR224, iImageNet_A, iFood101, iOmnibenchmark, iCUB200, iImageNet_R
from torch.utils.data import DataLoader
import torch
import random
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import time
from tqdm import tqdm
import json

def get_datasets(name, args):
    if name == 'cifar224':
        return iCIFAR224(args=args)
    elif name == 'imageneta':
        return iImageNet_A(args=args)
    elif name == 'ifood101':
        return iFood101(args=args)
    elif name == 'omnibenchmark':
        return iOmnibenchmark(args=args)
    elif name == 'cub':
        return iCUB200(args=args)
    elif name == 'imagenetr':
        return iImageNet_R(args=args)
    else:
        raise ValueError('Unknown dataset')


class MyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(self.images[idx], torch.Tensor):
            image = self.images[idx]
        else:
            image = self.trsf(Image.fromarray(np.uint8(self.images[idx])))
        label = self.labels[idx]
        return idx, image, label


class DataManger(Dataset):
    def __init__(self, dataset_name, device, args):
        self.device = device
        self.args = args
        self.init_class = self.args['init_class']
        self.increment = self.args['increment']
        dataset = get_datasets(dataset_name, args)
        # set learning order
        if args['shuffle']:
            self.class_order = list(dataset.class_order)
            np.random.seed(args['seed'])
            np.random.shuffle(self.class_order)
        else:
            self.class_order = dataset.class_order
        dataset.data_initialization()
        self.train_trsf = transforms.Compose(dataset.train_trsf)
        self.test_trsf = transforms.Compose(dataset.test_trsf)
        self.task_size, self.learning_list = self.setup_data()
        if self.args['dataset'] == 'cifar224':
            self.train_data = (dataset.train_data, dataset.train_targets)
            self.test_data = (dataset.test_data, dataset.test_targets)
            self.category_index = dataset.category_index
        else:
            self.train_data = dataset.train_data
            self.test_data = dataset.test_data
            self.category_index = dataset.category_index

    def map_order2cat(self, order):
        cat = self.class_order[order]
        return cat

    def map_cat2order(self, cat):
        order = self.class_order.index(cat)
        return order

    def map_cat2cat_name(self, cat):
        cat_name = self.category_index[cat]
        return cat_name

    def setup_data(self):
        if self.increment == 0:
            task_size = 1
        else:
            task_size = (len(self.class_order) - self.init_class) // self.increment
        learning_list = self.class_order[:self.init_class]
        for i in range(task_size - 1):
            learning_list.append(
                self.class_order[self.init_class + self.increment * i:self.init_class + self.increment * (i + 1)])
        return task_size, learning_list

    def get_task_list(self, task_id):
        if task_id == 0:
            train_list = self.class_order[:self.init_class]
            test_list = train_list
        else:
            train_list = self.class_order[
                         self.init_class + (task_id - 1) * self.increment:self.init_class + (task_id) * self.increment]
            test_list = self.class_order[:self.init_class + (task_id) * self.increment]
        train_list_name = [self.map_cat2cat_name(train_list[i]) for i in range(len(train_list))]
        return train_list, test_list, train_list_name

    def get_task_data(self, source: str, class_list: list):
        if source == 'train':
            source_data = self.train_data
            trsf = self.train_trsf
        elif source == 'test':
            source_data = self.test_data
            trsf = self.test_trsf
        elif source == 'train_no_aug':
            source_data = self.train_data
            trsf = self.test_trsf
        else:
            raise ValueError('Unknown source')
        if self.args['dataset'] == 'cifar224':
            data = []
            label = []
            for i in range(len(source_data[1])):
                if source_data[1][i] in class_list:
                    data.append(source_data[0][i])
                    label.append(source_data[1][i])
            if len(data) == 0:
                raise ValueError('No data')
        else:
            data_index = []
            for i in class_list:
                data_index.extend(source_data[i])
            data, label = split_images_labels(data_index)
        return MyDataset(data, label, trsf=trsf)


def split_images_labels(imgs):
    mode = '1'
    if mode == '0':
        images = []
        labels = []
        for img in tqdm(imgs):
            images.append(np.array(Image.open(img[0]).convert('RGB')))
            labels.append(int(img[1]))
        return images, np.array(labels)
    elif mode == '1':
        t0 = time.time()
        pfunc = partial(get_pil_img)
        pool = Pool(processes=12)
        results = pool.map(pfunc, imgs)
        pool.close()
        pool.join()
        images, labels = zip(*results)
        t1 = time.time()
        print("Pool takes %.2f s" % (t1 - t0))
    else:
        raise ValueError('Unknown mode')
    return images, np.array(labels)

def get_pil_img(imgs):
    with open(imgs[0], "rb") as f:
        img = Image.open(f).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img)
    label = imgs[1]
    return img, label

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param