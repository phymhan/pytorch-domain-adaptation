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
from data import MNISTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
    dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                    transform=Compose([GrayscaleToRgb(), ToTensor()]))
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
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


def do_epoch_bnn(model, dataloader, criterion, optim=None, T=1):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)

        # y_prob = 0.
        # for t in range(T):
        #     y_pred, s_pred = model(x)
        #     y_prob += 1./T * F.softmax(reparameterize(y_pred, s_pred), dim=1)
        # loss = F.nll_loss(torch.log(y_prob), y_true)
        # # True Bayesian network should average over probabilities (T times),
        # # however, logit with CrossEntropyLoss is more stable than log(softmax) with NLLLoss
        y_logit = 0.
        for t in range(T):
            y_pred, s_pred = model(x)
            y_logit += 1./T * reparameterize(y_pred, s_pred)
        loss = criterion(y_logit, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_logit.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

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
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)


def main_bnn(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)
    if args.test_target:
        target_dataset = MNISTM(train=False)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    else:
        target_loader = None
    model = BayesNet().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch_bnn(model, train_loader, criterion, optim=optim, T=args.T)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch_bnn(model, val_loader, criterion, optim=None, T=args.T)

        if args.test_target:
            tar_loss, tar_accuracy = do_epoch_bnn(model, target_loader, criterion, optim=None, T=args.T)
            tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                       f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, val_loss={tar_loss:.4f}, tar_accuracy={tar_accuracy:.4f}')
        else:
            tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                       f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source_bnn.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    arg_parser.add_argument('--bnn', action='store_true')
    arg_parser.add_argument('--T', type=int, default=1)
    arg_parser.add_argument('--test-target', action='store_true')
    args = arg_parser.parse_args()
    if args.bnn:
        main_bnn(args)
    else:
        main(args)
