import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GSBlock
from .nb_module import MeanAct, DispAct
from .transformer import AttnBlock, SelectItem


def _random_grayscale(x, p: float = 0.1):
    if torch.rand(1, device=x.device) >= p:
        return x
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.repeat(1, 3, 1, 1)


def _random_horizontal_flip(x, p: float = 0.2):
    if torch.rand(1, device=x.device) >= p:
        return x
    return torch.flip(x, dims=[3])


def _random_rotation(x, degrees: float = 90.0):
    n, _, _, _ = x.shape
    angles = (torch.rand(n, device=x.device) * 2.0 - 1.0) * degrees
    angles = angles * math.pi / 180.0

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    theta = torch.zeros((n, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _augment_patches(x):
    x = _random_grayscale(x, p=0.1)
    x = _random_rotation(x, degrees=90.0)
    x = _random_horizontal_flip(x, p=0.2)
    return x


class ConvmixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.dw(x) + x
        x = self.pw(x)
        return x


class MixerTransformer(nn.Module):
    def __init__(
        self,
        channel=32,
        kernel_size=5,
        dim=1024,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.0,
        policy="mean",
        gcn=True,
    ):
        super().__init__()
        self.layer1 = nn.Sequential(*[ConvmixerBlock(channel, kernel_size) for _ in range(depth1)])
        self.layer2 = nn.Sequential(
            *[AttnBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth2)]
        )
        self.layer3 = nn.ModuleList([GSBlock(dim, dim, policy, gcn) for _ in range(depth3)])
        self.jknet = nn.Sequential(
            nn.LSTM(dim, dim, 2),
            SelectItem(0),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x, ct, adj):
        x = self.down(self.layer1(x))
        g = x.unsqueeze(0)
        g = self.layer2(g + ct).squeeze(0)
        jk = []
        for layer in self.layer3:
            g = layer(g, adj)
            jk.append(g.unsqueeze(0))
        g = torch.cat(jk, 0)
        g = self.jknet(g).mean(0)
        return g


class ViT(nn.Module):
    def __init__(
        self,
        channel=32,
        kernel_size=5,
        dim=1024,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
        policy="mean",
        gcn=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = MixerTransformer(
            channel,
            kernel_size,
            dim,
            depth1,
            depth2,
            depth3,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            policy,
            gcn,
        )

    def forward(self, x, ct, adj):
        x = self.dropout(x)
        x = self.transformer(x, ct, adj)
        return x


class Hist2ST(nn.Module):
    def __init__(
        self,
        learning_rate=1e-5,
        fig_size=112,
        label=None,
        dropout=0.2,
        n_pos=64,
        kernel_size=5,
        patch_size=7,
        n_genes=785,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=16,
        channel=32,
        zinb=0,
        nb=False,
        bake=0,
        lamb=0,
        policy="mean",
    ):
        super().__init__()
        dim = (fig_size // patch_size) ** 2 * channel // 8
        self.learning_rate = learning_rate

        self.nb = nb
        self.zinb = zinb

        self.bake = bake
        self.lamb = lamb

        self.label = label
        self.patch_embedding = nn.Conv2d(3, channel, patch_size, patch_size)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(
            channel=channel,
            kernel_size=kernel_size,
            heads=heads,
            dim=dim,
            depth1=depth1,
            depth2=depth2,
            depth3=depth3,
            mlp_dim=dim,
            dropout=dropout,
            policy=policy,
            gcn=True,
        )
        self.channel = channel
        self.patch_size = patch_size
        self.n_genes = n_genes
        if self.zinb > 0:
            if self.nb:
                self.hr = nn.Linear(dim, n_genes)
                self.hp = nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake > 0:
            self.coef = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )

    def forward(self, patches, centers, adj, aug=False):
        '''
            patch: local tissue, RGB images
            centers: spatial cooridnate of each spot
        '''
        
        b, n, c, h, w = patches.shape
        patches = patches.reshape(b * n, c, h, w)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        ct = centers_x + centers_y
        h = self.vit(patches, ct, adj)
        x = self.gene_head(h)
        extra = None
        if self.zinb > 0:
            if self.nb:
                r = self.hr(h)
                p = self.hp(h)
                extra = (r, p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra = (m, d, p)
        if aug:
            h = self.coef(h)
        # return x, extra, h
        return x

    def aug(self, patch, center, adj):
        bake_x = []
        for _ in range(self.bake):
            new_patch = _augment_patches(patch.squeeze(0)).unsqueeze(0)
            x, _, h = self(new_patch, center, adj, True)
            bake_x.append((x.unsqueeze(0), h.unsqueeze(0)))
        return bake_x

    def distillation(self, bake_x):
        new_x, coef = zip(*bake_x)
        coef = torch.cat(coef, 0)
        new_x = torch.cat(new_x, 0)
        coef = F.softmax(coef, dim=0)
        new_x = (new_x * coef).sum(0)
        return new_x
