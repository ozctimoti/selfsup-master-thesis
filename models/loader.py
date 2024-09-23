import torch
import glob

from PIL import Image
from torch.utils.data import Dataset

from transforms import *

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """
    def __init__(self, root, pipeline):
        super(ContrastiveDataset, self).__init__()

        self.imgs = glob.glob(root + '*.jpg')
        self.pipeline = pipeline

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path).convert("RGB")

        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        im_q = self.pipeline(img)
        im_k = self.pipeline(img)

        return im_q, im_k

class ContrastiveDatasetw_Coords(Dataset):
    def __init__(self, root, distortions):
        self.distortions = distortions
        self.imgs = glob.glob(root + '*.jpg')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path).convert("RGB")

        img_q, coords_q = transform_w_coord(img, self.distortions)
        img_k, coords_k = transform_w_coord(img, self.distortions)
        return img_q, img_k, coords_q, coords_k


class ContrastiveDatasetWeakAugmentation(Dataset):
    def __init__(self, root, weak_aug, strong_aug):
        super(ContrastiveDatasetWeakAugmentation, self).__init__()

        self.imgs = glob.glob(root + '*.jpg')

        self.weak_aug = weak_aug
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # img = self.data_source.get_sample(idx)
        path = self.imgs[idx]
        img = Image.open(path).convert("RGB")

        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        im_q = self.strong_aug(img)
        im_k = self.weak_aug(img)

        return im_q, im_k


