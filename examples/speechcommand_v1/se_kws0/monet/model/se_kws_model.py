import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from monet.model.se_kws import SEKWS

def init_model(configs):
    
    se_kws_model = DCCRN(configs)

    return se_kws_model


