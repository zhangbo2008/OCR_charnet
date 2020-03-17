# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
# pip install yacs
from yacs.config import CfgNode as CN


_C = CN()

_C.INPUT_SIZE = 2280
_C.SIZE_DIVISIBILITY = 1
_C.WEIGHT= ""

_C.CHAR_DICT_FILE = ""
_C.WORD_LEXICON_PATH = ""

_C.WORD_MIN_SCORE = 0.95
_C.WORD_NMS_IOU_THRESH = 0.15
_C.CHAR_MIN_SCORE = 0.25
_C.CHAR_NMS_IOU_THRESH = 0.3
_C.MAGNITUDE_THRESH = 0.2

_C.WORD_STRIDE = 4
_C.CHAR_STRIDE = 4
_C.NUM_CHAR_CLASSES = 68

_C.WORD_DETECTOR_DILATION = 1
_C.RESULTS_SEPARATOR = chr(31)
# chr() 用一个范围在 range（256）内的（就是0～255）整数作参数，返回一个对应的字符。



# 把这个训练集的参数都写这里就行了.
import sys,os
_C.INPUT_SIZE=2280
_C.WEIGHT= "c:/icdar2015_hourglass88.pth"
tmp=os.path.abspath("..")+'/'
print(tmp)
_C.CHAR_DICT_FILE=tmp+"datasets/ICDAR2015/test/char_dict.txt"
_C.WORD_LEXICON_PATH=tmp+"datasets/ICDAR2015/test/GenericVocabulary.txt"
_C.RESULTS_SEPARATOR= ","
_C.SIZE_DIVISIBILITY= 128






print(_C.RESULTS_SEPARATOR)