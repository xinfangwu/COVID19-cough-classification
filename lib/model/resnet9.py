import torch.nn as nn
from torchvision import  models
import torch 

def conv_block(in_channels, out_channels, pool=False,p_size=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(p_size))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, use_pretrained=False):
        super().__init__()

        self.conv1 = conv_block(in_channels, 8)                             #32x150x150
        self.conv2 = conv_block(8, 16, pool=True,p_size=4)                  #64x37x37
        self.res1 = nn.Sequential(conv_block(16, 16), conv_block(16, 16))

        self.conv3 = conv_block(16, 32, pool=True,p_size=4)                 #128x9x9
        self.conv4 = conv_block(32, 64, pool=True,p_size=4)                #256x2x2
        self.res2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),                     #256x1x1
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, xb):

        out = self.conv1(xb)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.dropout(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = self.res2(out) + out

        out = self.classifier(out)

        return out

'''
class ResNet9(nn.Module):
    def __init__(self, num_classes=2, input_channel= 1 , num_layer=18, use_pretrained=False):
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
        self.model = nn.Sequential(*list(self.model.children())[:-3], 
                list(self.model.children())[-2], 
                #nn.Linear(128, self.num_classes),
                #list(self.model.children())[-1]
                )

               
        # final fc -> binary in_features=512 
        #self.model[7] = nn.Linear(1280, self.num_classes)


    def forward(self, x):
        output = self.model(x)
        output = torch.squeeze(output).cuda()
        m = nn.Linear(in_features=output.size()[1], out_features=2).cuda()
        output = m(output).cuda()
        return output
            
'''    
if __name__ == '__main__':
    model = ResNet9()
    print(model)
