import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """Simple CNN baseline over per-spot image patches."""

    def __init__(
        self,
        n_genes: int = 785,
        in_channels: int = 3,
        width: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        channels = [width * (2**i) for i in range(depth)]
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            prev = ch

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev, n_genes)

    def forward(self, patches, positions=None, adj=None):
        if isinstance(patches, (list, tuple)):
            patches = patches[0]

        if patches.ndim == 4:
            patches = patches.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        b, n, c, h, w = patches.shape
        x = patches.reshape(b * n, c, h, w)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        x = x.reshape(b, n, -1)

        if squeeze_batch:
            return x.squeeze(0)
        return x
