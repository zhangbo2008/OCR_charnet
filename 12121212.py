
from charnet.config import cfg
print( cfg.WORD_LEXICON_PATH)
print( cfg.INPUT_SIZE)
print( cfg.WORD_STRIDE)


import numpy as np
x=np.zeros([3,6,7])
y=lambda x: x[[ 0]]
print(y(x).shape)


