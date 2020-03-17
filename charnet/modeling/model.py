# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from charnet.modeling.backbone.resnet import resnet50
from charnet.modeling.backbone.hourglass import hourglass88
from charnet.modeling.backbone.decoder import Decoder
from collections import OrderedDict
from torch.functional import F
from charnet.modeling.layers import Scale
import torchvision.transforms as T
from .postprocessing import OrientedTextPostProcessing
from charnet.config import cfg


def _conv3x3_bn_relu(in_channels, out_channels, dilation=1):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU())
    ]))

#  传入*表示解包, 如果tensors是数组,也能接受.
def to_numpy_or_none(*tensors):
    results = []
    for t in tensors:
        if t is None:
            results.append(None)
        else:
            results.append(t.cpu().numpy())
    return results


# 下面4个类,表示的网络就是整个论文核心了.

class WordDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, dilation=1):
        super(WordDetector, self).__init__()
        self.word_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels, dilation
        )# bottleneck_channels 就是out_channel的意思.
        self.word_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.word_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.word_det_conv_final(x)

        pred_word_fg = self.word_fg_pred(self.word_fg_feat(feat))

        word_regression_feat = self.word_regression_feat(feat)
        pred_word_tblr = F.relu(self.word_tblr_pred(word_regression_feat)) * 10.
        pred_word_orient = self.orient_pred(word_regression_feat)


#  下面看看这个3个return的表示什么物理含义.
        # 他们结构是2,4,1
        # 第一个2 中的第一个数表示 是word的概率,第二个数表示不是word的概率
        # 4 : box坐标
        # orient:一个数表示角度.
        return pred_word_fg, pred_word_tblr, pred_word_orient


class CharDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, curved_text_on=False):
        super(CharDetector, self).__init__()
        self.character_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels
        )
        self.char_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.char_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)

    def forward(self, x):
        feat = self.character_det_conv_final(x)

        pred_char_fg = self.char_fg_pred(self.char_fg_feat(feat))
        char_regression_feat = self.char_regression_feat(feat)
        pred_char_tblr = F.relu(self.char_tblr_pred(char_regression_feat)) * 10.
        pred_char_orient = None
# return 的shape 还是2,4,1  含义跟上面一个class一样,只不过这里面orient强制给None了.
        return pred_char_fg, pred_char_tblr, pred_char_orient


class CharRecognizer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, num_classes):
        super(CharRecognizer, self).__init__()

        self.body = nn.Sequential(
            _conv3x3_bn_relu(in_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
        )
        self.classifier = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        feat = self.body(feat)
        return self.classifier(feat)











class CharNet(nn.Module):
    def __init__(self, backbone=hourglass88()):
        super(CharNet, self).__init__()
        self.backbone = backbone
        decoder_channels = 256
        bottleneck_channels = 128

        self.word_detector = WordDetector(
            decoder_channels, bottleneck_channels,
            dilation=cfg.WORD_DETECTOR_DILATION
        )

        self.char_detector = CharDetector(
            decoder_channels,
            bottleneck_channels
        )

        self.char_recognizer = CharRecognizer(
            decoder_channels, bottleneck_channels,
            num_classes=cfg.NUM_CHAR_CLASSES
        )

# 后处理的参数表.
        args = {
            "word_min_score": cfg.WORD_MIN_SCORE,
            "word_stride": cfg.WORD_STRIDE,
            "word_nms_iou_thresh": cfg.WORD_NMS_IOU_THRESH,
            "char_stride": cfg.CHAR_STRIDE,
            "char_min_score": cfg.CHAR_MIN_SCORE,
            "num_char_class": cfg.NUM_CHAR_CLASSES,
            "char_nms_iou_thresh": cfg.CHAR_NMS_IOU_THRESH,
            "char_dict_file": cfg.CHAR_DICT_FILE,
            "word_lexicon_path": cfg.WORD_LEXICON_PATH
        }
# 下面2个是后处理函数.
        self.post_processing = OrientedTextPostProcessing(**args)
# 這個是前處理函數.
        self.transform = self.build_transform()
# 核心是forward.整个算法核心.
    def forward(self, im, im_scale_w, im_scale_h, original_im_w, original_im_h):
        # im = self.transform(im).cuda()  # 因為這裡面有cuda,所以我電腦估計跑不了這個.
        im = self.transform(im)  # 因為這裡面有cuda,所以我電腦估計跑不了這個.
        # 原因就是pytorch的 cpu權重和 gpu權重不兼容.
        im = im.unsqueeze(0)   # 补充一个首位的batch_size 位置.保证shape一致.

        # 得到特征图.
        features = self.backbone(im)






# 下面就在特征图上进行各种操作了.还是conv而已. 得到7个张亮.
        pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features)
        pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features)
        recognition_results = self.char_recognizer(features)  # 68个char分类的得分.



# 注意下面进行softmax 必须dim=1, 因为dim0 是batch_size. 运算之后就得到了概率.其实就是让大的更大,小的更小.
        pred_word_fg = F.softmax(pred_word_fg, dim=1)
        pred_char_fg = F.softmax(pred_char_fg, dim=1)
        pred_char_cls = F.softmax(recognition_results, dim=1)







# 只是转化成numpy而已.
        pred_word_fg, pred_word_tblr, \
        pred_word_orient, pred_char_fg, \
        pred_char_tblr, pred_char_cls, \
        pred_char_orient = to_numpy_or_none(
            pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            pred_char_orient
        )
# 下面就进入核心的后处理过程.前处理没啥特殊的,就是基本fastercnn类似.




        # 发现一个问题,这个代码没有训练过程啊!!!!!!!!!!!!!!!!!!!!1
        # 最核心的loss函数不知道他怎么计算的!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        char_bboxes, char_scores, word_instances = self.post_processing(
            pred_word_fg[0, 1], pred_word_tblr[0],
            pred_word_orient[0, 0], pred_char_fg[0, 1],
            pred_char_tblr[0], pred_char_cls[0],
            im_scale_w, im_scale_h,
            original_im_w, original_im_h
        )

        return char_bboxes, char_scores, word_instances

    def build_transform(self):
        # 下行表示吧数组第一维度的index 2,1,0提取出来重新拼上.难道颜色对不上?????????
        # 处理的不是r,g,b这个顺序么?要改成b,g,r?  bgr模式才是这个代码需要用的编码.
        to_rgb_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(), #  (C x H x W)  这个是图片的shape信息.
                to_rgb_transform,
                normalize_transform,
            ]
        )
        return transform
