import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from data import MNISTM, ImageClassdata
import torchvision.transforms as transforms
from models import Net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.adapt_setting == 'svhn2mnist':
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
    elif args.adapt_setting == 'mnist2usps':
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
    else:
        raise NotImplementedError
    dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=1, pin_memory=True)

    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(dataloader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_accuracy = total_accuracy / len(dataloader)
    print(f'Accuracy on target data: {mean_accuracy:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=256)
    arg_parser.add_argument('--adapt-setting', type=str, default='svhn2mnist')
    arg_parser.add_argument('--tar-root', type=str, default=None)
    arg_parser.add_argument('--tar-list', type=str, default=None)
    arg_parser.add_argument('--img-type', type=str, default='RGB')
    args = arg_parser.parse_args()
    main(args)
