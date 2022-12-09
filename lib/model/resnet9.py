import torch.nn as nn
from torchvision import  models

class ResNet9(nn.Module):
    def __init__(self, num_classes=2, input_channel= 1 , num_layer=18, use_pretrained=True):
        super(ResNet9, self).__init__()
        
        self.num_classes = num_classes
        self.num_channel = input_channel
        
        self.num_layer = num_layer

        if self.num_layer == 18:
            self.model = models.resnet18(pretrained=use_pretrained)
        
        # adjust the first layer of model
        self.model.conv1 = nn.Conv2d(self.num_channel,
                                        64,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False)

        # drop half layers
        self.model = nn.Sequential(*list(self.model.children())[:-4], list(self.model.children())[-2], list(self.model.children())[-1])
        
        # final fc -> binary in_features=512 
        self.model[7] = nn.Linear(128, self.num_classes)


    def forward(self, x):
        output = self.model(x)
        return output
            
    
if __name__ == '__main__':
    model = ResNet9()
    print(model)
