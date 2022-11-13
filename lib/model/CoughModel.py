import torch.nn as nn

from .simplenet import SimpleNet
from .resnet import ResNet

class CoughModel(nn.Module):
    def __init__(self,num_classes=2, input_ch= 4, classify_ch=18):
        super(CoughModel, self).__init__()
        self.num_classes = num_classes
        self.encoder = SimpleNet(num_input_feat=input_ch, num_output_feat=classify_ch)
        self.classifymodel = ResNet(input_channel=classify_ch)

    def forward(self, x_spec, x_wave):
        feat_spec = self.encoder.forward(x_spec)
        feat_wave = self.encoder.forward(x_wave)
        output = self.classifymodel(feat_spec, feat_wave)

        return output



if __name__ == '__main__':
    model = CoughModel()
    print(model)


