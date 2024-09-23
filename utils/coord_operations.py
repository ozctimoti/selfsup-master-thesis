import torch

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
            in_q.append(torch.Tensor([i, (xmin - cq[0])/(cq[2] - cq[0]), (ymin - cq[1])/(cq[3] - cq[1]),
                                      1 - (cq[2] - xmax)/(cq[2] - cq[0]), 1 - (cq[3] - ymax)/(cq[3] - cq[1])]))

            in_k.append(torch.Tensor([(xmin - ck[0])/(ck[2] - ck[0]), (ymin - ck[1])/(ck[3] - ck[1]),
                                      1 - (ck[2] - xmax)/(ck[2] - ck[0]), 1 - (ck[3] - ymax)/(ck[3] - ck[1])]))

    return torch.stack(in_q), torch.stack(in_k)