import torch
import torch.nn as nn
import os
import sys

import torch.nn.functional as F
from torchmetrics import MeanSquaredLogError

from monet.model.se_model import init_model as se_model
from wekws.model.kws_model import init_model as kws_model

class SEKWS(nn.Module):
    def __init__(self, se_configs, kws_configs):

        self.se_configs = se_configs
        self.kws_configs = kws_configs

        self.se_model = se_model(se_configs)
        self.kws_model = kws_model(kws_configs)


    def forward(self, noisy_signal):

        _, enhanced_signal = self.se_model(noisy_signal)

        logits, _ = self.kws_model(enhanced_signal)

        return logits
