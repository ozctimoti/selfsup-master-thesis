import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import roi_align

from transforms import *
def is_flipped(coords):
    return coords[2] < coords[0]


def intersection(coords_q, coords_k):
    in_q, in_k = [], []
    for i, (cq, ck) in enumerate(zip(coords_q, coords_k)):
        if (ck[2] < cq[0] or cq[2] < ck[0]) or (ck[3] < cq[1] or cq[3] < ck[1]):
            in_k.append(torch.Tensor([0., 0., 1., 1.]))
        else:
            if ck[0] <= cq[0] <= ck[2]:
                xmin = cq[0]
            else:
                xmin = ck[0]

            if ck[0] <= cq[2] <= ck[2]:
                xmax = cq[2]
            else:
                xmax = ck[2]

            if ck[1] <= cq[1] <= ck[3]:
                ymin = cq[1]
            else:
                ymin = ck[1]

            if ck[1] <= cq[3] <= ck[3]:
                ymax = cq[3]
            else:
                ymax = ck[3]
            in_q.append(torch.Tensor([i, (xmin - cq[0]) / (cq[2] - cq[0]), (ymin - cq[1]) / (cq[3] - cq[1]),
                                      1 - (cq[2] - xmax) / (cq[2] - cq[0]), 1 - (cq[3] - ymax) / (cq[3] - cq[1])]))

            in_k.append(torch.Tensor([(xmin - ck[0]) / (ck[2] - ck[0]), (ymin - ck[1]) / (ck[3] - ck[1]),
                                      1 - (ck[2] - xmax) / (ck[2] - ck[0]), 1 - (ck[3] - ymax) / (ck[3] - ck[1])]))

    return torch.stack(in_q), torch.stack(in_k)

def area(coords):
    w = coords[2] - coords[0]
    h = coords[3] - coords[1]
    print(w*h)


'''
img_tensor = torch.randn((3, 800, 640))
img = T.ToPILImage()(img_tensor)

distortion = [
    RandomResizedCropWithLocation(size=(224, 224), scale=(0.08, 1.)),
    T.RandomApply([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                      )], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
    RandomHorizontalFlipReturnsIfFlip(p=1.),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

im_q, c_q = transform_w_coord(img, distortion)
im_k, c_k = transform_w_coord(img, distortion)

c_q = torch.unsqueeze(c_q, dim=0)
c_k = torch.unsqueeze(c_k, dim=0)

# print(c_q, c_k)

with torch.no_grad():
    c_q = [torch.Tensor([c[2], c[1], c[0], c[3]]) if is_flipped(c) else c for idx, c in enumerate(torch.unbind(c_q))]
    c_q = torch.stack(c_q)

    c_k = [torch.Tensor([c[2], c[1], c[0], c[3]]) if is_flipped(c) else c for idx, c in enumerate(torch.unbind(c_k))]
    c_k = torch.stack(c_k)

# print(c_q, c_k)

for idx, c in enumerate(torch.unbind(c_q)):
    area(c)

for idx, c in enumerate(torch.unbind(c_k)):
    area(c)

c_q, c_k = intersection(c_q, c_k)#
# print(c_q, c_k)
'''

# _, coords_q = transform_w_coord(img, distortion)
# coords = torch.rand((2, 4)) * 7
c_q = torch.Tensor([
    [0.3, 0.2, 0.5, 0.6],
    [0.1, 0.3, 0.9, 0.6],
    [0.1, 0.3, 0.5, 0.6],
    [0, 0, 1, 1]
])

c_k = torch.Tensor([
    [0.8, 0.2, 0.2, 0.7],
    [0.2, 0.4, 0.5, 0.5],
    [0.6, 0.4, 0.8, 0.5],
    [1/3, 0, 0, 1/3]
])

# print(coords)
with torch.no_grad():
    c_q = [torch.Tensor([c[2], c[1], c[0], c[3]]) if is_flipped(c) else c for idx, c in enumerate(torch.unbind(c_q))]
    c_q = torch.stack(c_q)

    c_k = [torch.Tensor([c[2], c[1], c[0], c[3]]) if is_flipped(c) else c for idx, c in enumerate(torch.unbind(c_k))]
    c_k = torch.stack(c_k)

    for idx, c in enumerate(torch.unbind(c_q)):
        area(c)

    print()
    for idx, c in enumerate(torch.unbind(c_k)):
        area(c)

    c_q, c_k = intersection(c_q, c_k)
    print(c_q)
    print()
    print(c_k)

    idx = c_q.transpose(0, 1)[0].long()
    print(c_k[idx])
    # print(c_k)
    # print(c_q.size())

f_q = torch.randn((4, 4, 3, 3))
f_k = torch.randn((4, 4, 3, 3))
out = roi_align(f_q, c_q, 1, spatial_scale=3, aligned=True)
print()
print(out.size())
print('----')
out_s = roi_align(f_q, c_q, (3, 3), spatial_scale=3, aligned=True)
out_flipped = out_s.flip(-1)

avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
out_s = avgpool(out_s).view(out_s.size(0), -1)
out_flipped = avgpool(out_flipped).view(out_s.size(0), -1)
print(out)
print(out_s)
print(out_flipped)
print()

out2 = roi_align(f_k, [c_k], 1, spatial_scale=3, aligned=True)

print(out2.size())
idx = c_q.transpose(0, 1)[0].long()
print(idx)
print(out2[idx].size())

# print(out)
# print(out.size())
# print('********')
# print(out_2)
# b = [torch.flip(t, [2]) if is_flipped_[idx] else t for idx, t in enumerate(torch.unbind(f_q))]
# x = torch.stack(b)

# x = torch.flip(f_q, [3])

# [number if number > 30 else 0 for number in numbers]

