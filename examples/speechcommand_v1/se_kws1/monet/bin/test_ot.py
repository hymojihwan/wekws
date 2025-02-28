from __future__ import print_function

import argparse
import copy
import logging
import os
import matplotlib.pyplot as plt
import multiprocessing
import json
import time
from datetime import timedelta, datetime

from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from monet.model.se_kws_ot_model import SEKWSModel
from monet.utils.file_utils import setup_logging
from monet.utils.train_utils import count_parameters, set_mannul_seed, get_last_checkpoint, display_epoch_progress
from monet.utils.checkpoint import load_checkpoint, save_checkpoint
from monet.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Joint test for SE + KWS')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument("--test_data", required=True, help="Path to the test data JSON")
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
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--log_file", default="test.log", help="File to save logs")


    args = parser.parse_args()
    return args


def acc_kws(logits, target):
    """ Compute accuracy for Keyword Spotting """
    pred = logits.argmax(dim=1)  # 가장 높은 확률을 가진 클래스 선택
    correct = pred.eq(target).sum().item()
    return correct  # 🔥 배치 크기 고려


def setup_logging(log_file):
    """ Set up logging to file and console """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def test(model, dataloader, device):
    """
    모델 검증 함수
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            key, noisy, clean, target = batch 
            clean, noisy, target = clean.to(device), noisy.to(device), target.to(device)

            _, enhanced_audio = model.se_model(noisy)
            enhanced_feats, _ = model.feature_extracter(enhanced_audio)
            logits, _ = model.kws_model(enhanced_feats)
            
            total_correct += acc_kws(logits, target)
            total_samples += len(target)

    # 🔥 Final Accuracy Calculation
    avg_acc = (total_correct / total_samples) * 100

    return avg_acc


def main():
    args = get_args()
    # Set up logging
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Init asr model from configs
    model = SEKWSModel(configs['se_model'], configs['kws_model'])

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, 'accuracy_result.txt')

    # 결과 파일 초기화
    with open(result_file, 'w') as fout:
        fout.write("Accuracy Results:\n")
        fout.write("=================\n\n")

    # 테스트 데이터 로드
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['feature_extraction_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.test_data, test_conf)
    test_data_loader = DataLoader(test_dataset,
                                    batch_size=None,
                                    pin_memory=args.pin_memory,
                                    num_workers=args.num_workers)

    # 테스트 실행
    with torch.no_grad():
        test_results = test(model, test_data_loader, device)

    # 결과를 파일에 저장
    logging.info(f"🚀 Final Accuracy Results 🚀")
    logging.info(f"Noisy Speech Accuracy: {test_results:.2f}%")  # ✅ 수정
    
    # Save final results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(json.dumps(test_results, indent=4))

    logging.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
