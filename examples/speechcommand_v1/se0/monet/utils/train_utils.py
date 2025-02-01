#!/usr/bin/env python3
# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import random
import re, os


def set_mannul_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_last_checkpoint(model_dir):
    """
    Finds the last saved checkpoint in the given directory.
    
    Args:
        model_dir (str): Path to the directory containing checkpoint files.
    
    Returns:
        Tuple[int, str]: Tuple containing the epoch number and the checkpoint path.
    """
    checkpoints = [
        f for f in os.listdir(model_dir) if re.match(r'^\d+\.pt$', f)
    ]
    if not checkpoints:
        return -1, None  # No checkpoints found

    # Sort checkpoints by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('.')[0]))
    last_checkpoint = checkpoints[-1]
    last_epoch = int(last_checkpoint.split('.')[0])
    return last_epoch, os.path.join(model_dir, last_checkpoint)