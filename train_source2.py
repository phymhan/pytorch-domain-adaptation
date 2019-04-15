import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from models import Net, BayesNet
from utils import GrayscaleToRgb, reparameterize
import torch.nn.functional as F
from data import MNISTM, ImageClassdata
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(args):
    if args.adapt_setting == 'svhn2mnist':
        source_dataset = ImageClassdata(txt_file=args.src_list, root_dir=args.src_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
    elif args.adapt_setting == 'mnist2usps':
        source_dataset = ImageClassdata(txt_file=args.src_list, root_dir=args.src_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
    else:
        raise NotImplementedError
    train_loader = DataLoader(source_dataset, batch_size=args.batch_size, drop_last=True,
                              shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(target_dataset, batch_size=args.batch_size, drop_last=False,
                            shuffle=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        # print(x[0,0,...])
        # print(y_true)
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):
    train_loader, val_loader = create_dataloaders(args)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'trained_models/{args.adapt_setting}_{args.name}.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    arg_parser.add_argument('--T', type=int, default=1)
    arg_parser.add_argument('--test-target', action='store_true')
    arg_parser.add_argument('--name', type=str, default='sourcemodel')
    arg_parser.add_argument('--adapt-setting', type=str, default='svhn2mnist')
    arg_parser.add_argument('--src-root', type=str, default=None)
    arg_parser.add_argument('--tar-root', type=str, default=None)
    arg_parser.add_argument('--src-list', type=str, default=None)
    arg_parser.add_argument('--tar-list', type=str, default=None)
    arg_parser.add_argument('--img-type', type=str, default='RGB')
    args = arg_parser.parse_args()
    main(args)
