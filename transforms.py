import math
import random
import torch

from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from PIL import ImageFilter

'''Useless TwoCropTransform'''
'''
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
'''

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomResizedCropWithLocation(torch.nn.Module):
    def __init__(
        self,
        size=(224, 224),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        # super().__init__()
        super(RandomResizedCropWithLocation, self).__init__()

        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, H, W = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / W, float(i) / H, float(j + w) / W, float(i + h) / H])
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord


class RandomHorizontalFlipReturnsIfFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        # super().__init__()
        super(RandomHorizontalFlipReturnsIfFlip, self).__init__()
        self.p = p

    def forward(self, img, coords):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            coords (List): Coordinate of the image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
            List: Coordinate of randomly flipped image
        """
        if torch.rand(1) < self.p:
            coords_tr = coords.clone()
            coords_tr[0] = coords[2] # in pixpro; pixpro coord format is different.
            coords_tr[2] = coords[0]
            return F.hflip(img), coords_tr
        return img, coords

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

def transform_w_coord(img, distortions):
    coords = None
    for distortion in distortions:
        if type(distortion) == RandomResizedCropWithLocation:
            img, coords = distortion(img)
        elif type(distortion) == RandomHorizontalFlipReturnsIfFlip:
            img, coords = distortion(img, coords)
        else:
            img = distortion(img)

    return img, coords
