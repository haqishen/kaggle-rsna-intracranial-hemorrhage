import sys
sys.path.append('/data/src/EfficientNet-PyTorch')

import torch.nn as nn

from efficientnet_pytorch import model as enet


class enetv2(nn.Module):
    def __init__(self, backbone, out_dim=6):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.myfc(dropout(x))
            else:
                h += self.myfc(dropout(x))
        return h / len(self.dropouts)
