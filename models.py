import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class DensNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.pre_classifier = nn.Sequential(nn.Linear(1024, 256, bias=True),
                                            nn.BatchNorm1d(256),nn.ReLU())
        self.classifier = nn.Linear(256, num_classes, bias=True)

        del preloaded

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.pre_classifier(out)
        return self.classifier(out),out