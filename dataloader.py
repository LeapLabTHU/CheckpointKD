# -*- coding: utf-8 -*-
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
import os
import time

def get_train_loader(data, data_root,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    train iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: train set iterator.
    """
    
    if data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))

    elif data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))

    else:
        # ImageNet
        traindir = os.path.join(data_root, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))


    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )
    # for i, (images, labels) in enumerate(train_loader):
    #     print(labels)
    #     print(labels.size())
    #     time.sleep(3)

    return train_loader

def get_test_loader(data, data_root,
                    batch_size,
                    num_workers=4,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
   
    if data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        val_set = datasets.CIFAR10(data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        val_set = datasets.CIFAR100(data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))

    else:
        # ImageNet
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))


    data_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # for i, (images, labels) in enumerate(data_loader):
    #     print(labels)
    #     print(labels.size())
    #     time.sleep(3)

    return data_loader



''' 
    # define transforms
    trans = transforms.Compose([
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR100(
        data_dir, train=False, download=False, transform=trans
    )
'''
'''    
    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 将图像转化为32 * 32
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # 归一化
    ])
    # [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

    # load dataset
    dataset = datasets.CIFAR100(root=data_dir,
                                transform=trans,
                                download=False,
                                train=True)
'''
    