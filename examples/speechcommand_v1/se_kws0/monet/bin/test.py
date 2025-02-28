# Pre-trained SE model & KWS model just load and inference

import os
import logging
import argparse
import copy
import torch
import torch.nn.functional as F
import json
import torchaudio

import time
import yaml
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from monet.dataset.dataset import Dataset
from monet.model.se_model import init_model as init_se_model
from monet.model.kws_model import init_model as init_kws_model
from monet.utils.checkpoint import load_checkpoint

# 🔥 CUDA 문제 방지: 'spawn' start method 설정
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # 이미 설정된 경우 무시

def get_args():
    parser = argparse.ArgumentParser(description="Testing SE + KWS Model")
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
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
    parser.add_argument("--config_se", required=True, help="Path to SE model config file")
    parser.add_argument("--config_kws", required=True, help="Path to KWS model config file")
    parser.add_argument("--checkpoint_se", required=True, help="Path to SE model checkpoint")
    parser.add_argument("--checkpoint_kws", required=True, help="Path to KWS model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to the test data JSON")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--log_file", default="test.log", help="File to save logs")
    parser.add_argument("--num_keywords", required=True, help="How many predict keywords")
    args = parser.parse_args()
    return args


def setup_logging(log_file):
    """ Set up logging to file and console """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def load_test_data(test_data_path):
    """ Load test data from a JSON-formatted list file """
    test_data = []
    with open(test_data_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                test_data.append(entry)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid line in test data: {line.strip()}")
    return test_data


def acc_kws(logits, target):
    """ Compute accuracy for Keyword Spotting """
    pred = logits.argmax(dim=1)  # 가장 높은 확률을 가진 클래스 선택
    correct = pred.eq(target).sum().item()
    return correct  # 🔥 배치 크기 고려


def test_se_kws(kws_model, test_data_loader, device):
    """ Evaluate SE + KWS Model and measure latency """

    kws_model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            batch_keys, batch_feats, batch_targets, _, _ = batch
            batch_feats = batch_feats.to(device)  # KWS 입력
            batch_targets = batch_targets.to(device)  # 정답 라벨

            # 🔥 Latency 측정 시작
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            logits, _ = kws_model.forward(batch_feats)  # ✅ KWS 모델 추론 수행

            # 🔥 Latency 측정 종료
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            total_correct += acc_kws(logits, batch_targets)
            total_samples += len(batch_targets)

    # 🔥 Final Accuracy Calculation
    avg_acc = (total_correct / total_samples) * 100

    return avg_acc, total_samples

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Set up logging
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path)

    # Load SE Model
    with open(args.config_se, "r") as f:
        se_configs = yaml.load(f, Loader=yaml.FullLoader)
    se_model = init_se_model(se_configs["model"])
    load_checkpoint(se_model, args.checkpoint_se)

    # Load KWS Model
    with open(args.config_kws, "r") as f:
        kws_configs = yaml.load(f, Loader=yaml.FullLoader)
    kws_model = init_kws_model(kws_configs["model"])
    load_checkpoint(kws_model, args.checkpoint_kws)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_model = se_model.to(device)
    kws_model = kws_model.to(device)

    test_conf = copy.deepcopy(kws_configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['feature_extraction_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size

    start_time = time.time()
    # Load test data
    test_dataset = Dataset(args.test_data, se_model, test_conf)
    test_data_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers)

    # Run evaluation
    test_results, total_samples = test_se_kws(kws_model, test_data_loader, device)
    end_time = time.time()
    avg_latency = (end_time - start_time) * 1000  # ms 단위 변환
    avg_latency = avg_latency / total_samples
    # Log final results
    logging.info(f"🚀 Final Accuracy Results 🚀")
    logging.info(f"Noisy Speech Accuracy: {test_results:.2f}%")
    logging.info(f"Average Latency per Batch: {avg_latency:.2f} ms")  # ✅ 추가

    # Save final results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, "w") as f:
        json.dump({"accuracy": test_results, "avg_latency_ms": avg_latency}, f, indent=4)

    logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()