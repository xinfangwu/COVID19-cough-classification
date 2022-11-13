import torch 
import torch.nn as nn
from torchvision import  models

class ResNet(nn.Module):
    def __init__(self, num_classes=2, input_channel= 64 , num_layer=152, use_pretrained=True):
        super(ResNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_channel = input_channel
        
        assert num_layer in [18, 34, 50, 101, 152]
        self.num_layer = num_layer

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
        
        # adjust the first layer of model
        self.model.conv1 = nn.Conv2d(self.num_channel * 2,
                                        64,
                                        kernel_size=(7, 7),
                                        stride=(2, 2),
                                        padding=(3, 3),
                                        bias=False)
        # final fc -> binary 
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


    def forward(self, x1, x2):
        # input = (128,128,64) & (128,128,64) -> (128, 128, 128)
        x = torch.concat((x1, x2), 1)
        output = self.model(x)
        return output
            
    
if __name__ == '__main__':
    model = ResNet()
    print(model)
