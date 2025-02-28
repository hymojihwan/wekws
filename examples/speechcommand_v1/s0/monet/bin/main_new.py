import argparse
import copy
import logging
import os
import time
import multiprocessing
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.functional as F
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from monet.model.bc_resnet import BCResNets
from monet.utils.file_utils import setup_logging
from monet.utils.train_utils import (
    count_parameters, set_mannul_seed, get_last_checkpoint, display_epoch_progress
)
from monet.utils.checkpoint import load_checkpoint, save_checkpoint
from monet.dataset.dataset import SpeechCommand, Padding  # 🔥 새로운 데이터셋 추가
# from speech_command import SpeechCommand, Padding  # 🔥 새로운 데이터셋 추가


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpus',
                        default='-1',
                        help='gpu lists, seperated with `,`, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=True,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn_file', default=None, help='global cmvn file')
    parser.add_argument('--norm_var',
                        action='store_true',
                        default=False,
                        help='norm var option')
    parser.add_argument('--num_keywords',
                        default=1,
                        type=int,
                        help='number of keywords')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    set_mannul_seed(args.seed)

    # Config 파일 로드
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Multi-GPU 설정
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    gpu = int(args.gpus.split(',')[rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    if world_size > 1:
        logging.info(f'Training on multiple GPUs, current GPU: {gpu}')
        dist.init_process_group(backend=args.dist_backend)

    # Logging 설정
    log_file = os.path.join(args.model_dir, 'train.log')
    setup_logging(log_file, rank)

    # Dataset 설정 🔥 새로운 데이터셋 적용
    train_dataset = SpeechCommand(args.train_data, ver=1, transform=Padding())
    cv_dataset = SpeechCommand(args.cv_data, ver=1, transform=Padding())

    train_data_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    cv_data_loader = DataLoader(
        cv_dataset, batch_size=100, num_workers=args.num_workers
    )

    input_dim = configs['dataset_conf']['feature_extraction_conf'][
        'n_mels']
    output_dim = args.num_keywords
    configs['model']['output_dim'] = output_dim
    if 'input_dim' not in configs['model']:
        configs['model']['input_dim'] = input_dim
    
    # 모델 설정
    model = BCResNets(int(configs['model']['tau']))
    num_params = count_parameters(model)
    logging.info(f"Model Parameters: {num_params}")

    # Checkpoint 자동 로드
    last_epoch, last_checkpoint = get_last_checkpoint(args.model_dir)
    if args.checkpoint:
        infos = load_checkpoint(model, args.checkpoint)
    elif last_checkpoint:
        logging.info(f"Loading checkpoint: {last_checkpoint}")
        infos = load_checkpoint(model, last_checkpoint)
        start_epoch = last_epoch + 1
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        infos = {}

    start_epoch = infos.get('epoch', 0) + 1
    num_epochs = configs['training_config'].get('max_epoch', 100)

    train_losses = infos.get('train_losses', [])
    cv_losses = infos.get('cv_losses', [])

    # 학습 루프
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        logging.info(f"Epoch {epoch} TRAINING STARTED")
        for sample in tqdm(train_data_loader, desc="epoch %d, iters" % (epoch)):
            inputs, labels = sample
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_loss = F.cross_entropy(outputs, labels.long())
            train_loss.backward()
            optimizer.step()
            model.zero_grad()
        logging.info(f"Epoch Training Loss: {train_loss:.4f}")
    logging.info("Training Finished.")

if __name__ == '__main__':
    main()
