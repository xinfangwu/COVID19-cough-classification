import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, num_input_feat=4, num_output_feat=64 ):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            # (2048, 2048, 4) -> (1024, 1024, 64)
            nn.Conv2d(num_input_feat, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        )
        self.layer2 = nn.Sequential(
            # (1024, 1024, 64) -> (512, 512, 256)
            nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        )
        self.layer3 = nn.Sequential(
            # (512, 512, 256) -> (256, 256, 128)
            nn.Conv2d(256, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        )
        self.layer4 = nn.Sequential(
            # (256, 256, 128) -> (128, 128, 64)
            nn.Conv2d(128, num_output_feat, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_output_feat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == '__main__':
    model = SimpleNet()
    print(model)