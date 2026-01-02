import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(cfg):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    dataset_class = torchvision.datasets.CIFAR10
    transform_train = torchvision.transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.RandomCrop(cfg.aug_rand_crop_size, padding=cfg.aug_rand_crop_padding),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=cfg.aug_rand_num_ops, magnitude=cfg.aug_rand_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = dataset_class(
        root="./datasets", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    testset = dataset_class(
        root="./datasets", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return trainloader, testloader, testloader
