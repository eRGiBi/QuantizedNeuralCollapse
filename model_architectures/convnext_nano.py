import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.drop_path = (
        #     DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtNano(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, drop_path_rate=0.0):
        super().__init__()

        # Opt 15M params: [64, 128, 256, 512] channels, [2, 2, 6, 2] depths
        dims = [64, 128, 256, 512]
        depths = [2, 2, 6, 2]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNormChannelsFirst(dims[0], eps=1e-6),
        )

        # Downsampling layers
        self.downsample_layers = nn.ModuleList([self.stem])
        for i in range(3):
            downsample = nn.Sequential(
                LayerNormChannelsFirst(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # Stages with stochastic depth
        # self.stages = nn.ModuleList()
        # dp_rates = [
        #     x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        # ]
        # cur = 0
        # for i in range(4):
        #     stage = nn.Sequential(
        #         *[Block(dims[i], dp_rates[cur + j]) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Block(dims[i], drop_path=0.0) for j in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # global average pooling
        x = self.norm(x)
        x = self.head(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == "__main__":
#     model = ConvNeXtTiny15M()
#     print(count_parameters(model))
#
#     x = torch.randn(1, 3, 224, 224)
#     out = model(x)
#     print(f"input {x.shape}")
#     print(f"output {out.shape}")