from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from skimage import io, transform
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision import transforms, utils
from matplotlib  import pyplot as plt


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))
        label = resize(label, (new_h, new_w))
        return {'image': img, 'label': label}

class Rotate(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            if random.random() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)        
            else:
                image = np.flipud(image)
                label = np.flipud(label)        

        return {'image': image,
                'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()
       
        return {'image': image.unsqueeze(0),
                'label': label}

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        #image = (image - image.min())/(image.max()-image.min())
        image = image- image.mean() 
        if label.max() != label.min():
            label = (label - label.min())/(label.max()-label.min())
        return {'image': image, 'label': label}

class ProstateDataset(Dataset):

    def __init__(self, csv_file, phase, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        image = np.load(img_name)
        label_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 1])
        label = np.load(label_name)

        sample = {'image': image, 'label': label}
        if self.transform:
        	sample = self.transform(sample)
        return sample
