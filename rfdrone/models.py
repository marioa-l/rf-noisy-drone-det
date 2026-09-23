"""VGG models for 2D spectrogram input and 1D IQ input.

The 2D variant mirrors lib/model_VGG2D.py of the authors' repository exactly
(attribute names `features`, `avgpool`, `classifier` and layer order), so the
state_dict of the pretrained VGG11_BN published on Zenodo loads into it.
The 1D variant follows the NCTA 2023 paper: Conv2d -> Conv1d,
MaxPool2d -> MaxPool1d, AdaptiveAvgPool2d -> AdaptiveAvgPool1d.
"""
import torch
import torch.nn as nn

CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, cfg, dim=2, batch_norm=True, num_classes=7, in_channels=2, hidden=256):
        super().__init__()
        conv, bn, pool, apool = {
            1: (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d),
            2: (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d),
        }[dim]
        layers, c = [], in_channels
        for v in cfg:
            if v == "M":
                layers.append(pool(kernel_size=2, stride=2))
                continue
            layers.append(conv(c, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(bn(v))
            layers.append(nn.ReLU(inplace=True))
            c = v
        self.features = nn.Sequential(*layers)
        self.avgpool = apool(1)
        self.classifier = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(True), nn.Dropout(), nn.Linear(hidden, num_classes))
        self._init_weights()

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))

    def embed(self, x):
        """Activations of the 256-unit dense layer (used for embedding plots)."""
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.classifier[1](self.classifier[0](x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def build_model(name, dim, num_classes, in_channels=2):
    """name: vgg11 | vgg11_bn | vgg13 | ... ; dim: 1 (IQ) or 2 (spectrogram)."""
    base, batch_norm = (name[:-3], True) if name.endswith("_bn") else (name, False)
    return VGG(CFGS[base], dim=dim, batch_norm=batch_norm, num_classes=num_classes,
               in_channels=in_channels)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
