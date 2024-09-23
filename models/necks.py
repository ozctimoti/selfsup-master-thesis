import torch.nn as nn
import torchvision

__all__ = ['densecl_neck', 'deco2v2_neck']

class DenseCLNeck(nn.Module):
    def __init__(self, in_channels=2048, hid_channels=2048, out_channels=128):
        super(DenseCLNeck, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels)
        )
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.avgpool(x) # (N, C, 1, 1)
        logits = self.mlp(logits.view(logits.size(0), -1))  # (N, C')

        x = self.dense(x) # (N, C'', S, S)

        avgpooled_x2 = self.avgpool2(x) # (N, C'', 1, 1)
        x = x.view(x.size(0), x.size(1), -1) # (N, C'', S^2)

        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # (N, C'')
        return [logits, x, avgpooled_x2]  # (N, C'), (N, C'', S^2), (N, C'')


class DeCo2v2Neck(nn.Module):
    def __init__(self, in_channels=2048, hid_channels=2048, out_channels=128):
        super(DeCo2v2Neck, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels)
        )
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, c_x):
        # logits = torchvision.ops.roi_align(x, c_x, 1, spatial_scale=x.size(-1), aligned=True)
        logits = torchvision.ops.roi_align(x, c_x, x.size(0), spatial_scale=x.size(-1), aligned=True)
        logits = self.avgpool(logits)
        logits = self.mlp(logits.view(logits.size(0), -1))  # (N, C')

        x = self.dense(x) # (N, C'', S, S)

        avgpooled_x2 = self.avgpool2(x) # (N, C'', 1, 1)
        x = x.view(x.size(0), x.size(1), -1) # (N, C'', S^2)

        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # (N, C'')
        return [logits, x, avgpooled_x2]  # (N, C'), (N, C'', S^2), (N, C'')

def densecl_neck(in_channels=2048, hid_channels=2048, out_channels=128, **kwargs):
    return DenseCLNeck(in_channels, hid_channels, out_channels, **kwargs)

def deco2v2_neck(in_channels=2048, hid_channels=2048, out_channels=128, **kwargs):
    return DeCo2v2Neck(in_channels, hid_channels, out_channels, **kwargs)