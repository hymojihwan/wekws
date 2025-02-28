import os
import logging
import random
import argparse
import torch
import torch.distributed as dist
import json
import torchaudio
import yaml
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader, Dataset

import monet.dataset.processor as processor
from wekws.utils.file_utils import read_lists

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = data.copy()
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data

class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        lists = self.sampler.sample(self.lists)
        for src in lists:
            # yield dict(src=src)
            data = dict(src=src)
            data.update(sampler_info)
            yield data

def Dataset(test_data, conf, reverb_lmdb=None, noise_lmdb=None):
    lists = read_lists(test_data)

    # 🔥 기존 DataList를 확장하여 SE 모델 적용 후 KWS 데이터 생성
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=True)
    dataset = Processor(dataset, processor.parse_raw)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    # dataset = Processor(dataset, processor.padding_sample)
    return dataset