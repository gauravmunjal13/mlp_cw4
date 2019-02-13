import numpy as np
import os
import torchvision.transforms as transforms
import utils

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder

def normalize(x_train, x_test):
    num_channels = x_train.shape[1]
    for i in range(num_channels):
        X_mean = x_train[:, i, :, :].mean()
        X_sd = x_train[:, i, :, :].std()
        x_train[:, i, :, :] -= X_mean
        x_train[:, i, :, :] /= X_sd
        x_test[:, i, :, :] -= X_mean
        x_test[:, i, :, :] /= X_sd
    return x_train, x_test

def save_bird_or_bicycle():
    print('Saving bird or bicycle')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder('data/bird_or_bicycle/extras', transform=transform)
    x_train = []
    y_train = []
    for x_batch, y_batch in DataLoader(train_dataset):
        x_train.append(x_batch)
        y_train.append(y_batch)
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    test_dataset = ImageFolder('data/bird_or_bicycle/test', transform=transform)
    x_test = []
    y_test = []
    for x_batch, y_batch in DataLoader(test_dataset):
        x_test.append(x_batch)
        y_test.append(y_batch)
    x_test = np.concatenate(x_test)
    y_test = np.array(y_test)
    x_train, x_test = normalize(x_train, x_test)
    x_train, y_train, x_test, y_test = x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), \
        y_test.astype('float32')
    utils.save_file((x_train, y_train, x_test, y_test), 'data/bird_or_bicycle/bird_or_bicycle.pkl')

def save_cifar10():
    print('Saving cifar10')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10('data/cifar10', train=True, transform=transform)
    x_train = []
    y_train = []
    for x_batch, y_batch in DataLoader(train_dataset):
        x_train.append(x_batch)
        y_train.append(y_batch)
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    test_dataset = CIFAR10('data/cifar10', train=False, transform=transform)
    x_test = []
    y_test = []
    for x_batch, y_batch in DataLoader(test_dataset):
        x_test.append(x_batch)
        y_test.append(y_batch)
    x_test = np.concatenate(x_test)
    y_test = np.array(y_test)
    x_train, x_test = normalize(x_train, x_test)
    x_train, y_train, x_test, y_test = x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), \
        y_test.astype('float32')
    utils.save_file((x_train, y_train, x_test, y_test), 'data/cifar10/cifar10.pkl')

def get_data(args):
    fpath = f'data/{args.dataset_name}/{args.dataset_name}.pkl'
    if not os.path.exists(fpath):
        if args.dataset_name == 'cifar10':
            save_cifar10()
        elif args.dataset_name == 'bird_or_bicycle':
            save_bird_or_bicycle()
        else:
            raise NotImplementedError
    return utils.load_file(fpath)