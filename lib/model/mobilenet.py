import torch.nn as nn
from torchvision import  models


class MobileNet(nn.Module):
    def __init__(self, num_classes=2, input_channel=1, use_pretrained=True):
        super(MobileNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_channel = input_channel

        self.model = models.mobilenet_v2(pretrained=use_pretrained)

        # let the model get gray scale imge
        self.model.features[0][0] = nn.Conv2d(self.num_channel,
                                                    32,  #n_filters
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    bias=False)

        # final fc -> binary 
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)


    def forward(self, x):
        output = self.model(x)
        return output
            