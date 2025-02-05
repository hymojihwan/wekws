import os
import logging
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


class SEKWS_DataList(DataList):
    """
    SE 모델을 적용한 후 KWS 모델용 데이터로 변환하는 DataList
    """
    def __init__(self, lists, se_model, shuffle=True, partition=True):
        super().__init__(lists, shuffle, partition)
        self.se_model = se_model

    def __iter__(self):
        sampler_info = self.sampler.update()
        lists = self.sampler.sample(self.lists)

        for src in lists:
            try:
                json_line = json.loads(src)
                assert 'key' in json_line and 'wav' in json_line and 'txt' in json_line

                key = json_line['key']
                wav_path = json_line['wav']
                label = json_line['txt']

                # 🔥 Noisy Speech Load (CPU에서 실행)
                noisy_signal, sr = torchaudio.load(wav_path)

                # 🔥 SE 모델 적용 (GPU에서만 실행)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        noisy_signal = noisy_signal.to(device)
                        se_model = self.se_model.to(device)
                    else:
                        device = torch.device("cpu")
                        se_model = self.se_model.to(device)

                    _, enhanced_signal = self.se_model(noisy_signal)
                
                # 🔥 JSON 대신 Tensor 자체를 넘김
                yield {
                    "key": key,
                    "wav": enhanced_signal.cpu(),  # 🔥 바로 텐서로 넘김
                    "sample_rate": sr,
                    "txt": label
                }

            except Exception as e:
                logging.warning(f"Skipping invalid data: {src} ({str(e)})")

def Dataset(test_data, se_model, conf, reverb_lmdb=None, noise_lmdb=None):
    lists = read_lists(test_data)

    # 🔥 기존 DataList를 확장하여 SE 모델 적용 후 KWS 데이터 생성
    shuffle = conf.get('shuffle', True)
    dataset = SEKWS_DataList(lists, se_model, shuffle=shuffle, partition=True)
    dataset = Processor(dataset, processor.parse_raw)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)
    if reverb_lmdb and conf.get('reverb_prob', 0) > 0:
        reverb_data = LmdbData(reverb_lmdb)
        dataset = Processor(dataset, processor.add_reverb,
                            reverb_data, conf['reverb_prob'])
    if noise_lmdb and conf.get('noise_prob', 0) > 0:
        noise_data = LmdbData(noise_lmdb)
        dataset = Processor(dataset, processor.add_noise,
                            noise_data, conf['noise_prob'])
    feature_extraction_conf = conf.get('feature_extraction_conf', {})
    if feature_extraction_conf['feature_type'] == 'mfcc':
        dataset = Processor(dataset, processor.compute_mfcc,
                            **feature_extraction_conf)
    elif feature_extraction_conf['feature_type'] == 'fbank':
        dataset = Processor(dataset, processor.compute_fbank,
                            **feature_extraction_conf)
    spec_aug = conf.get('spec_aug', True)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)

    context_expansion = conf.get('context_expansion', False)
    if context_expansion:
        context_expansion_conf = conf.get('context_expansion_conf', {})
        dataset = Processor(dataset, processor.context_expansion,
                            **context_expansion_conf)

    frame_skip = conf.get('frame_skip', 1)
    if frame_skip > 1:
        dataset = Processor(dataset, processor.frame_skip, frame_skip)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset