import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_class_metrics(pred: list, label: list):
    # 整体识别率
    overall_accuracy = np.round(accuracy_score(label, pred), 4)

    # 获取所有类别
    unique_classes = np.unique(label)
    class_accuracies = {}
    class_confusion_matrices = np.zeros((len(unique_classes), len(unique_classes)))

    # 逐类别计算识别率和混淆矩阵
    for cls in unique_classes:
        cls_indices = np.where(np.array(label) == cls)[0]
        cls_correct = np.sum(np.array(pred)[cls_indices] == cls)
        class_accuracies[cls] = np.round(cls_correct / len(cls_indices))

    for i in range(len(pred)):
        class_confusion_matrices[pred[i], label[i]] += 1

    return {
        "all_accy": overall_accuracy,
        "class_accy": class_accuracies,
        "class_confusion_matrices": class_confusion_matrices,
    }


def calculate_task_metrics(pred, label, init_class, increment):
    for i in range(len(pred)):
        pred[i] = cat2task(pred[i], init_class, increment)
        label[i] = cat2task(label[i], init_class, increment)
    res = calculate_class_metrics(pred, label)
    all_accy, task_acc, task_confusion_matrices = res['all_accy'], res['class_accy'], res['class_confusion_matrices']
    return {
        "all_accy": all_accy,
        "task_accy": task_acc,
        "task_confusion_matrices": task_confusion_matrices,
    }


def cat2task(x, init_class, increment):
    if x < init_class:
        return 0
    else:
        max_task_size = (x - init_class) // increment + 1
        for i in range(1, max_task_size + 1):
            start_index, end_index = init_class, init_class + i * increment
            if start_index <= x < end_index:
                return i
    return None


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in tqdm(imgs):
        images.append(np.array(pil_loader(item[0])))
        labels.append(item[1])

    return np.array(images), np.array(labels)


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
