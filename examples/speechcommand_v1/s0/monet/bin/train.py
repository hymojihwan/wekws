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
from monet.dataset.dataset import Dataset

# from wekws.model.kws_model import init_model


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


# ✅ Loss 플로팅 함수 (비동기 실행)
def save_loss_plot(train_losses, cv_losses, model_dir, tag=""):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(cv_losses, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({tag})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_dir}/{tag}_loss.png")
    plt.close()

def plot_loss_async(train_losses, cv_losses, model_dir, tag=""):
    p = multiprocessing.Process(target=save_loss_plot, args=(train_losses, cv_losses, model_dir, tag))
    p.start()

def get_data_list_length(data_list_file):
    """ data.list 파일의 줄 개수를 세어 데이터셋 길이를 반환 """
    with open(data_list_file, "r") as f:
        return sum(1 for _ in f)  # 🔥 총 라인 개수 반환

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

    # Dataset 설정
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['shuffle'] = False
    cv_conf['spec_aug'] = False
    cv_conf['speed_perturb'] = False

    train_dataset = Dataset(args.train_data, train_conf)
    cv_dataset = Dataset(args.cv_data, cv_conf)

    train_data_loader = DataLoader(train_dataset, batch_size=None, pin_memory=args.pin_memory,
                                   num_workers=args.num_workers, prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset, batch_size=None, pin_memory=args.pin_memory,
                                num_workers=args.num_workers, prefetch_factor=args.prefetch)
    input_dim = configs['dataset_conf']['feature_extraction_conf'][
        'num_ceps']
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

    # Optimizer & Scheduler 설정
    # 🔥 Optimizer 선택 (SGD / Adam)
    optim_type = configs['optim']
    optim_conf = configs['optim_conf']

    init_lr = optim_conf['init_lr']
    lr_lower_limit = optim_conf['lr_lower_limit']
    lr = optim_conf['lr']
    weight_decay = optim_conf.get('weight_decay', 1e-3)
    momentum = optim_conf.get('momentum', 0.9)
    warmup_epochs = optim_conf.get('warmup_epochs', 5)
    num_samples = get_data_list_length(args.train_data)
    n_step_warmup = num_samples * warmup_epochs
    total_iter = num_samples * num_epochs
    iterations = 0

    if optim_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    model_dir = args.model_dir
    writer = None
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if world_size > 1:
        assert torch.cuda.is_available()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        device = torch.device("cuda")
    else:
        use_cuda = gpu >= 0 and torch.cuda.is_available()
        device = torch.device(f'cuda:{gpu}' if use_cuda else 'cpu')
        model = model.to(device)

    # 학습 루프
    start_time = time.time()
    cv_num_samples = get_data_list_length(args.cv_data)
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        logging.info(f"Epoch {epoch} TRAINING STARTED")
        total_epoch_loss = []
        for sample in tqdm(train_data_loader, desc="epoch %d, iters" % (epoch)):
            # lr cos schedule
            iterations += 1
            if iterations < n_step_warmup:
                lr = init_lr * iterations / n_step_warmup
            else:
                lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (1 + np.cos(np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            key, feats, target, _, _ = sample
            feats, target = feats.to(device), target.to(device)    
            outputs = model(feats)
            train_loss = F.cross_entropy(outputs, target.long())
            train_loss.backward()
            optimizer.step()
            model.zero_grad()

            total_epoch_loss.append(train_loss.item())


        epoch_loss = sum(total_epoch_loss) / len(total_epoch_loss)
        logging.info(f"Epoch Training Loss: {epoch_loss:.4f}")
        logging.info("current lr check ... %.4f" % lr)
        train_losses.append(epoch_loss)

        total_cv_epoch_loss = []
        with torch.no_grad():
            model.eval()
            true_count = 0.0
            num_valid_set = float(cv_num_samples)
            for key, feats, target, _, _ in cv_data_loader:
                feats, target = feats.to(device), target.to(device)   
                outputs = model(feats)
                valid_loss = F.cross_entropy(outputs, target.long())
                total_cv_epoch_loss.append(valid_loss.item())
                prediction = torch.argmax(outputs, dim=-1) 
                true_count += torch.sum(prediction == target).detach().cpu().numpy()
            acc = true_count / num_valid_set * 100
            valid_epoch_loss = sum(total_cv_epoch_loss) / len(total_cv_epoch_loss)
            logging.info(f"Epoch Validation Loss: {valid_epoch_loss:.4f}, acc : {acc:.2f}")
            cv_losses.append(valid_epoch_loss)
        # Checkpoint 저장
        save_checkpoint(model, os.path.join(args.model_dir, f"{epoch}.pt"), {
            'epoch': epoch,
            'train_losses': train_losses,
            'cv_losses': cv_losses,
        })

        plot_loss_async(train_losses, cv_losses, args.model_dir)
        display_epoch_progress(epoch, num_epochs, start_time)

    logging.info("Training Finished.")

if __name__ == '__main__':
    main()
