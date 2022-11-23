import torch
import torch.nn as nn
from torchvision import  models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class EncodeResNet(nn.Module):
    def __init__(self, num_input_feat=4, num_output_feat=64, num_layer=18, use_pretrained=True ):
        super(EncodeResNet, self).__init__()
        
        assert num_layer in [18, 34, 50, 101, 152]
        self.num_layer = num_layer
        self.num_input_feat = num_input_feat
        self.num_output_feat = num_output_feat
        
        if self.num_layer == 18:
            self.model = models.resnet18(pretrained=use_pretrained)
        elif num_layer == 34:
            self.model = models.resnet34(pretrained=use_pretrained)
        elif num_layer == 50:
            self.model = models.resnet50(pretrained=use_pretrained)
        elif num_layer == 101:
            self.model = models.resnet101(pretrained=use_pretrained)
        elif num_layer == 152:
            self.model = models.resnet152(pretrained=use_pretrained)


        # adjust the first layer
        self.model.conv1 = nn.Conv2d(self.num_input_feat,
                                        64,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False)
        
        # fc = self.model.fc.in_features
        
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        # self.model.avgpool = Identity()
        # self.model.fc = Identity()
        # add the last layer 
        self.layer5 = nn.Sequential(
            # (256, 256, 512) -> (128, 128, 64)
            nn.Conv2d(512, self.num_output_feat, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_output_feat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        x = self.model(x)
        x = self.layer5(x)
        return x


if __name__ == '__main__':
    model = EncodeResNet()
    print(model)
