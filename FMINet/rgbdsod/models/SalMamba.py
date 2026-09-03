import torch
import torch.nn as nn
import re
import torchvision.models as models
from torch.nn import functional as F
from rgbdsod.models.builder import EncoderDecoder
from rgbdsod.models.config_sunrgbd import config


class SalMamba(nn.Module):
    def __init__(self):
        super(SalMamba, self).__init__()
        self.model = EncoderDecoder(cfg=config).cuda()
        # print(self.model)

    def forward(self, rgb, depth):
        rgb=rgb.cuda()  # [1, 3, h, w]
        depth = depth.cuda()  # [1, 3, h, w]
        output=self.model(rgb, depth)
        return output


if __name__=="__main__":
    rgb=torch.randn([2, 3, 224,224])
    depth=torch.randn([2, 3, 224,224])

    model=SalMamba()

    output=model(rgb=rgb, depth=depth)

    print(output[0].shape)
