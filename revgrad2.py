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
from data import MNISTM, ImageClassdata
from models import Net, DTN, BDTN
from utils import GrayscaleToRgb, GradientReversal
import torchvision.transforms as transforms
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # TODO: add DTN model
    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    if args.adapt_setting == 'mnist2mnistm':
        source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                              transform=Compose([GrayscaleToRgb(), ToTensor()]))
        target_dataset = MNISTM(train=False)
    elif args.adapt_setting == 'svhn2mnist':
        source_dataset = ImageClassdata(txt_file=args.src_list, root_dir=args.src_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif args.adapt_setting == 'mnist2usps':
        source_dataset = ImageClassdata(txt_file=args.src_list, root_dir=args.src_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
        target_dataset = ImageClassdata(txt_file=args.tar_list, root_dir=args.tar_root, img_type=args.img_type,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                        ]))
    else:
        raise NotImplementedError
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))
    if not os.path.exists('logs'): os.makedirs('logs')
    f = open(f'logs/{args.adapt_setting}_{args.name}.txt', 'w+')

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

                features = feature_extractor(x).view(x.shape[0], -1)
                domain_preds = discriminator(features).squeeze()
                label_preds = clf(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()
                
                target_label_preds = clf(features[source_x.shape[0]:])
                target_label_accuracy += (target_label_preds.cpu().max(1)[1] == target_labels).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        target_mean_accuracy = target_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}, target_accuracy={target_mean_accuracy:.4f}')
        f.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                f'source_accuracy={mean_accuracy:.4f}, target_accuracy={target_mean_accuracy:.4f}\n')

        torch.save(model.state_dict(), f'trained_models/{args.adapt_setting}_{args.name}_ep{epoch}.pt')
    f.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    arg_parser.add_argument('--adapt-setting', type=str, default='svhn2mnist')
    arg_parser.add_argument('--src-root', type=str, default=None)
    arg_parser.add_argument('--tar-root', type=str, default=None)
    arg_parser.add_argument('--src-list', type=str, default=None)
    arg_parser.add_argument('--tar-list', type=str, default=None)
    arg_parser.add_argument('--img-type', type=str, default='RGB')
    arg_parser.add_argument('--name', type=str, default='revgrad2')
    args = arg_parser.parse_args()
    main(args)
