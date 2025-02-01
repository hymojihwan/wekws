# Copyright (c) 2021 Binbin Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import re

import yaml
import torch

def load_checkpoint(model: torch.nn.Module, path: str):
    logging.info(f'Loading checkpoint from {path}')
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)

    # 🔥 학습 정보 불러오기 (latest.yaml 우선 사용)
    yaml_path = os.path.join(os.path.dirname(path), "latest.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as fin:
            infos = yaml.load(fin, Loader=yaml.FullLoader)
        logging.info(f'Loaded training info from {yaml_path}')
    else:
        infos = {}

    return infos


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info(f'Checkpoint: saving to {path}')
    
    # 모델 state_dict 저장
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)

    # 🔥 항상 최신 정보는 "latest.yaml"에 저장
    info_path = os.path.join(os.path.dirname(path), "latest.yaml")
    if infos is None:
        infos = {}
    
    with open(info_path, 'w') as fout:
        yaml.dump(infos, fout)

    logging.info(f'Checkpoint info saved to {info_path}')