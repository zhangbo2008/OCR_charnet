# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
from collections import OrderedDict
from torch.functional import F

# 解码部分,只是一些conv运算.
class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(Decoder, self).__init__()
        #  初始化为了定义self.backbone_feature_reduction 和  self.top_down_feature_reduction
        self.backbone_feature_reduction = nn.ModuleList()
        self.top_down_feature_reduction = nn.ModuleList()
        #  好像传入的是一种金字塔图片,是一个图片组成的list. 应该用了fpn技术.
        #  list 是从高分辨率到低分辨率.
        for i, in_channels in enumerate(in_channels_list[::-1]):# 先逆序分辨率.
            self.backbone_feature_reduction.append(
                self._conv1x1_relu(in_channels, out_channels)
            )
            if i < len(in_channels_list) - 2: # 当i 不是最后2层时候.
                self.top_down_feature_reduction.append(
                    self._conv1x1_relu(out_channels, out_channels)
                )

    def _conv1x1_relu(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=1, # kernal=1的就是attention了.
                bias=False
            )),
            ("relu", nn.ReLU())
        ]))

    def forward(self, x):
        x = x[::-1]  # to lowest resolution first
        top_down_feature = None
        for i, feature in enumerate(x):
            # 首先输入的图片,进入backbone 里对应的index ,做里面的conv1*1运算.
            feature = self.backbone_feature_reduction[i](feature)

            if i == 0:
                top_down_feature = feature
            else:# 进行双现行差值,提升特征.
                # 吧上一次的低纬度的特征值进行bilinear差值运算,得到新的特征图.
                upsampled_feature = F.interpolate(
                    top_down_feature,
                    size=feature.size()[-2:], # 表示输出的大小. feature: channel*宽*高,所以这个size 等于feature的宽和高.
                    mode='bilinear',
                    align_corners=True
                )


                #  迭代过程中更新top_down_feature
                if i < len(x) - 1:
                    top_down_feature = self.top_down_feature_reduction[i - 1](
                        feature + upsampled_feature
                    )
                else:
                    top_down_feature = feature + upsampled_feature
        return top_down_feature
