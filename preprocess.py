import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder

def get_bird_or_bicycle_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = DataLoader(ImageFolder('data/bird_or_bicycle/extras', transform=transform), batch_size=batch_size,
        shuffle=True)
    test_data = DataLoader(ImageFolder('data/bird_or_bicycle/test', transform=transform), batch_size=batch_size)
    return train_data, test_data

def get_cifar10_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.491401, 0.482159, 0.446531], [0.247032, 0.243485, 0.261588])
    ])
    train_data = DataLoader(CIFAR10('data/cifar10', train=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(CIFAR10('data/cifar10', train=False, transform=transform), batch_size=batch_size)
    return train_data, test_data

def get_data(args):
    if args.dataset_name == 'cifar10':
        return get_cifar10_data(args.batch_size)
    elif args.dataset_name == 'bird_or_bicycle':
        return get_bird_or_bicycle_data(args.batch_size)
    else:
        raise ValueError