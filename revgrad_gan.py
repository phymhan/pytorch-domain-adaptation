"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
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
from models import Net
from utils import GrayscaleToRgb, GradientReversal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    discriminator = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    
    target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)

    optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optim_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        target_label_accuracy = 0
        for (source_x, source_labels), (target_x, target_labels) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                # forward
                features = feature_extractor(x).view(x.shape[0], -1)

                # train discriminator
                domain_preds = discriminator(features.detach()).squeeze()
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                optim_D.zero_grad()
                domain_loss.backward()
                optim_D.step()

                # train generator
                label_preds = clf(features[:source_x.shape[0]])
                label_loss = F.cross_entropy(label_preds, label_y)
                domain_preds = discriminator(features[:source_x.shape[0]]).squeeze()
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds,
                                                                 torch.zeros(source_x.shape[0]).to(device))
                loss = label_loss + domain_loss
                optim_G.zero_grad()
                loss.backward()
                optim_G.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()
                
                target_label_preds = clf(features[source_x.shape[0]:])
                target_label_accuracy += (target_label_preds.cpu().max(1)[1] == target_labels).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        target_mean_accuracy = target_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}, target_accuracy={target_mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/revgrad.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=15)
    arg_parser.add_argument('--lr', type=float, default=0.001)
    args = arg_parser.parse_args()
    main(args)
