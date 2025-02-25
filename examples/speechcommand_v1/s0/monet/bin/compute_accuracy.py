# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
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

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.dataset import Dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wekws.utils.executor import Executor


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', nargs='+', required=True, 
                        help='test data file (space-separated)')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--result_dir',
                        required=True, 
                        help='directory to save results') 

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Init asr model from configs
    model = init_model(configs['model'])

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    executor = Executor()
    model.eval()

    os.makedirs(args.result_dir, exist_ok=True)
    result_file = os.path.join(args.result_dir, 'accuracy_result.txt')

    # 결과 파일 초기화
    with open(result_file, 'w') as fout:
        fout.write("Accuracy Results:\n")
        fout.write("=================\n\n")

    for test_data in args.test_data:
        # 테스트 데이터 로드
        test_conf = copy.deepcopy(configs['dataset_conf'])
        test_conf['filter_conf']['max_length'] = 102400
        test_conf['filter_conf']['min_length'] = 0
        test_conf['speed_perturb'] = False
        test_conf['spec_aug'] = False
        test_conf['shuffle'] = False
        test_conf['feature_extraction_conf']['dither'] = 0.0
        test_conf['batch_conf']['batch_size'] = args.batch_size

        test_dataset = Dataset(test_data, test_conf)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=None,
                                      pin_memory=args.pin_memory,
                                      num_workers=args.num_workers)

        # 테스트 실행
        with torch.no_grad():
            test_loss, test_acc = executor.test(model, test_data_loader, device, configs['training_config'])

        # 결과를 파일에 저장
        dataset_name = os.path.basename(os.path.dirname(test_data))
        with open(result_file, 'a') as fout:
            fout.write(f"Dataset: {dataset_name}\n")
            fout.write(f"  Test Loss: {test_loss}\n")
            fout.write(f"  Test Accuracy: {test_acc}\n\n")

        logging.info(f"Dataset: {os.path.basename(test_data)}, Loss: {test_loss}, Accuracy: {test_acc}")

    logging.info(f"Results saved to {result_file}")




if __name__ == '__main__':
    main()
