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


def main(args):
    model = BayesNet().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(10, 50),
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

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

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

                # args.T should be 1 for current version
                y_preds, s_preds = model(x)
                logits = reparameterize(y_preds, s_preds)
                # logits = y_preds
                domain_preds = discriminator(s_preds).squeeze()
                # domain_preds = discriminator(logits).squeeze()
                label_preds = logits[:source_x.shape[0]]

                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()
                
                target_label_preds = logits[source_x.shape[0]:]
                target_label_accuracy += (target_label_preds.cpu().max(1)[1] == target_labels).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        target_mean_accuracy = target_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}, target_accuracy={target_mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/auda.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=15)
    arg_parser.add_argument('--T', type=int, default=1)
    args = arg_parser.parse_args()
    main(args)
