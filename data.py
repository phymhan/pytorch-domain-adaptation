import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
from PIL import Image

import config


class BSDS500(Dataset):

    def __init__(self):
        image_folder = config.DATA_DIR / 'BSR/BSDS500/data/images'
        self.image_files = list(map(str, image_folder.glob('*/*.jpg')))

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)


class MNISTM(Dataset):

    def __init__(self, train=True):
        super(MNISTM, self).__init__()
        self.mnist = datasets.MNIST(config.DATA_DIR / 'mnist', train=train,
                                    download=True)
        self.bsds = BSDS500()
        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)


class ImageClassdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, img_type, transform=transforms.ToTensor()):
        """
        Args:
            txt_fpred_conf_tensorile (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(txt_file, 'r') as f:
            self.images_frame = [l.strip('\n') for l in f.readlines()]
        self.root_dir = root_dir
        self.transform = transform
        self.img_type = img_type
        self.paths, self.labels = [], []
        for l in self.images_frame:
            self.paths.append(os.path.join(self.root_dir, l.split()[0]))
            self.labels.append(int(l.split()[1]))

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        if self.img_type == 'grayscale':
            image = Image.open(self.paths[idx])
        elif self.img_type == 'RGB':
            img = Image.open(self.paths[idx])
            image = img.convert('RGB')
        else:
            raise NotImplementedError

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
