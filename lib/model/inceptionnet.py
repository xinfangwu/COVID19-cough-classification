import torch.nn as nn
from torchvision import  models


class InceptionNetV3(nn.Module):
    def __init__(self, num_classes=2, input_channel=1, use_pretrained=True):
        super(InceptionNetV3, self).__init__()
        
        self.num_classes = num_classes
        self.num_channel = input_channel

        self.model = models.inception_v3(pretrained=use_pretrained)

        # let the model get gray scale imge
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(self.num_channel,
                                                        32,  #n_filters
                                                        kernel_size=(7, 7),
                                                        stride=(2, 2),
                                                        padding=(3, 3),
                                                        bias=False)

        # Handle the auxilary net
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, self.num_classes)
        # Handle the primary net
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    def forward(self, x):
        output = self.model(x)
        return output



    
    
    