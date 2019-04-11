"""
Adversarial Uncertainty Domain Adaptation
"""
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from data import MNISTM
from models import Net, BayesNet
from utils import GrayscaleToRgb, GradientReversal, reparameterize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-32


def main(args):
    global EPS
    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))

    discriminator = nn.Sequential(
        nn.Linear(10, 20),
        nn.LeakyReLU(0.2),
        nn.Linear(20, 20),
        nn.LeakyReLU(0.2),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    
    target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optim_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    gan_loss = torch.nn.BCELoss()

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        target_label_accuracy = 0
        for (source_x, source_labels), (target_x, target_labels) in tqdm(batches, leave=False, total=n_batches):
                source_x = source_x.to(device)
                source_labels = source_labels.to(device)
                target_x = target_x.to(device)

                # real = torch.ones(source_x.size(0), 1).to(device)
                # fake = torch.zeros(target_x.size(0), 1).to(device)

                # forward
                y_src = model(source_x)
                p_src = F.softmax(y_src, dim=1)
                I_src = -p_src * torch.log(p_src+EPS)
                y_tar = model(target_x)
                p_tar = F.softmax(y_tar, dim=1)
                I_tar = -p_tar * torch.log(p_tar+EPS)

                # print(I_src.size())
                # print(I_tar.size())

                # train discriminator
                optim_D.zero_grad()
                loss_D_src = gan_loss(discriminator(I_src.detach()), torch.ones(source_x.size(0), 1).to(device))
                loss_D_tar = gan_loss(discriminator(I_tar.detach()), torch.zeros(target_x.size(0), 1).to(device))
                loss_D = (loss_D_src + loss_D_tar) / 2
                loss_D.backward()
                optim_D.step()

                # train generator
                # real = torch.ones(target_x.size(0), 1).to(device)
                optim_G.zero_grad()
                loss_clf = F.cross_entropy(y_src, source_labels)
                loss_gan = gan_loss(discriminator(I_tar), torch.ones(target_x.size(0), 1).to(device))
                loss_G = loss_clf + loss_gan
                loss_G.backward()
                optim_G.step()

                total_domain_loss += loss_D.item()
                total_label_accuracy += (y_src.max(1)[1] == source_labels).float().mean().item()

                target_label_preds = y_tar
                target_label_accuracy += (target_label_preds.cpu().max(1)[1] == target_labels).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        target_mean_accuracy = target_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}, target_accuracy={target_mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/advent.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=15)
    arg_parser.add_argument('--T', type=int, default=1)
    arg_parser.add_argument('--lr', type=float, default=0.0002)
    args = arg_parser.parse_args()
    main(args)
