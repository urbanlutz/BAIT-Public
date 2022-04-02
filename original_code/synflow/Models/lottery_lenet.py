import torch
import torch.nn as nn
from original_code.synflow.Layers import layers

class LeNet5(nn.Module):
    def __init__(self, input_shape, num_classes, dense_classifier=False, pretrained=False):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            layers.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            layers.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            layers.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            layers.Linear(256, 256),
            nn.ReLU(inplace=True),
            layers.Linear(256, num_classes),
        )

        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)