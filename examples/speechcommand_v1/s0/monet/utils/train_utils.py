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
import time
import logging
from datetime import timedelta, datetime


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

def display_epoch_progress(epoch, total_epochs, start_time):
    """
    현재 Epoch 진행 상황과 예상 종료 시간을 출력하는 함수.
    
    Args:
        epoch (int): 현재 Epoch 번호
        total_epochs (int): 전체 Epoch 수
        start_time (float): 전체 학습 시작 시간 (time.time())
    """
    elapsed_time = time.time() - start_time
    avg_epoch_time = elapsed_time / epoch

    remaining_epochs = total_epochs - epoch
    remaining_time = avg_epoch_time * remaining_epochs
    estimated_end_time = datetime.now() + timedelta(seconds=remaining_time)

    logging.info(f"Epoch {epoch}/{total_epochs}")
    logging.info(f"Elapsed Time: {str(timedelta(seconds=int(elapsed_time)))}")
    logging.info(f"Estimated Remaining Time: {str(timedelta(seconds=int(remaining_time)))}")
    logging.info(f"Expected Completion Time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")