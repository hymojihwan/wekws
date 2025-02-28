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

import io
import logging
import json
import random

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'noisy' in obj
        assert 'clean' in obj
        key = obj['key']
        wav_file = obj['noisy']
        clean = obj['clean']
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
            target_wave, _ = torchaudio.load(clean)
            example = dict(key=key,
                           clean=target_wave,
                           noisy=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data, max_length=10240, min_length=10):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


def compute_mfcc(
    data,
    feature_type='mfcc',
    num_ceps=80,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
):
    """Extract mfcc

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(
            waveform,
            num_ceps=num_ceps,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def compute_fbank(data,
                  feature_type='fbank',
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=1000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


# def padding(data):
#     """ Padding the data into training data

#         Args:
#             data: Iterable[List[{key, clean, noisy}]]

#         Returns:
#             Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
#     """
    
#     for sample in data:
#         assert isinstance(sample, list)
#         noisy_length = torch.tensor([x['noisy'].size(0) for x in sample],
#                                     dtype=torch.int32)
        
#         order = torch.argsort(noisy_length, descending=True)
#         noisy_lengths = torch.tensor(
#             [sample[i]['noisy'].size(0) for i in order], dtype=torch.int32)
#         clean_length = torch.tensor([x['clean'].size(0) for x in sample],
#                                     dtype=torch.int32)
#         order_clean = torch.argsort(clean_length, descending=True)
#         clean_lengths = torch.tensor(
#             [sample[i]['clean'].size(0) for i in order_clean], dtype=torch.int32)
#         sorted_noisy = [sample[i]['noisy'] for i in order]
#         sorted_keys = [sample[i]['key'] for i in order]
#         sorted_clean = [sample[i]['clean'] for i in order_clean]
#         sorted_noisy = extend_list(sorted_noisy)
#         padded_noisy = pad_sequence(sorted_noisy,
#                                     batch_first=True,
#                                     padding_value=0)
#         sorted_clean = extend_list(sorted_clean)
#         padded_clean = pad_sequence(sorted_clean,
#                                     batch_first=True,
#                                     padding_value=0)
#         yield (sorted_keys, padded_noisy, padded_clean, noisy_lengths, clean_lengths)

def extend_list(lst):
    """
    길이가 16000보다 짧으면 반복해서 늘리고, 길면 잘라서 16000으로 맞춤.
    """
    sample_rate = 16000
    new_list = []

    for sample in lst:
        length = sample.size(-1)

        if length > sample_rate:
            # 너무 긴 경우 잘라서 맞춤
            new_sample = sample[:, :sample_rate]
        elif length < sample_rate:
            # 너무 짧은 경우 반복해서 확장
            repeat_factor = (sample_rate // length) + 1  # 필요한 반복 횟수 계산
            new_sample = sample.repeat(1, repeat_factor)[:, :sample_rate]  # 자르면서 확장
        else:
            new_sample = sample  # 이미 16000이면 그대로 유지
        
        new_list.append(new_sample)
    
    return new_list

def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, clean, noisy}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)

        # 각 데이터의 길이 확인
        noisy_lengths = torch.tensor([x['noisy'].size(0) for x in sample], dtype=torch.int32)
        clean_lengths = torch.tensor([x['clean'].size(0) for x in sample], dtype=torch.int32)

        # noisy 데이터 길이 기준으로 정렬
        order = torch.argsort(noisy_lengths, descending=True)
        sorted_noisy = [sample[i]['noisy'] for i in order]
        sorted_clean = [sample[i]['clean'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]

        # 동일한 랜덤 구간 선택 후 noisy, clean에 적용
        processed_noisy, processed_clean = [], []
        for noisy, clean in zip(sorted_noisy, sorted_clean):
            noisy_processed, clean_processed = process_pair(noisy, clean)
            processed_noisy.append(noisy_processed)
            processed_clean.append(clean_processed)
        
        # Zero-padding을 적용한 결과를 배치로 반환
        padded_noisy = pad_sequence(processed_noisy, batch_first=True, padding_value=0)
        padded_clean = pad_sequence(processed_clean, batch_first=True, padding_value=0)

        yield (sorted_keys, padded_noisy, padded_clean, noisy_lengths, clean_lengths)
        
def process_pair(noisy_waveform, clean_waveform, sample_rate=16000):
    """
    noisy와 clean 데이터를 동일한 구간으로 자르거나 패딩.

    Args:
        noisy_waveform (Tensor): (channels, time) 형태의 noisy 데이터
        clean_waveform (Tensor): (channels, time) 형태의 clean 데이터
        sample_rate (int): 샘플레이트 (기본값: 16000)

    Returns:
        Tuple(Tensor, Tensor): 동일한 구간으로 자른 noisy와 clean 데이터
    """
    FIXED_LENGTH = sample_rate * 1  # 4초 = 64000 샘플

    # 차원 확인 및 조정 (1D → 2D)
    if noisy_waveform.dim() == 1:
        noisy_waveform = noisy_waveform.unsqueeze(0)
    if clean_waveform.dim() == 1:
        clean_waveform = clean_waveform.unsqueeze(0)

    length = min(noisy_waveform.size(1), clean_waveform.size(1))

    if length > FIXED_LENGTH:
        # 동일한 랜덤 구간 선택
        start_idx = random.randint(0, length - FIXED_LENGTH)
        noisy_processed = noisy_waveform[:, start_idx:start_idx + FIXED_LENGTH]
        clean_processed = clean_waveform[:, start_idx:start_idx + FIXED_LENGTH]
    else:
        # Zero-padding 적용
        pad_length = FIXED_LENGTH - length
        noisy_padding = torch.zeros((noisy_waveform.size(0), pad_length), dtype=noisy_waveform.dtype)
        clean_padding = torch.zeros((clean_waveform.size(0), pad_length), dtype=clean_waveform.dtype)

        noisy_processed = torch.cat([noisy_waveform, noisy_padding], dim=-1)
        clean_processed = torch.cat([clean_waveform, clean_padding], dim=-1)

    return noisy_processed, clean_processed

# def extend_list(list):
#     sample_rate = 16000
#     new_list = []
#     for sample in list:
#         if sample.size(-1) != sample_rate:
#             # new_sample = sample
#             new_sample = torch.cat([sample, sample[:, :sample_rate-sample.size(-1)]], dim=-1)
#             for i in range(0, int(sample_rate / sample.size(-1)) -1):
#                 new_sample = torch.cat([new_sample, new_sample[:, :sample_rate-new_sample.size(-1)]], dim=-1)
#             new_list.append(new_sample)
#         else:
#             new_list.append(sample)
#         del sample
    
#     return new_list
    

def add_reverb(data, reverb_source, aug_prob):
    for sample in data:
        assert 'wav' in sample
        if aug_prob > random.random():
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            _, rir_data = reverb_source.random_one()
            rir_io = io.BytesIO(rir_data)
            _, rir_audio = wavfile.read(rir_io)
            rir_audio = rir_audio.astype(np.float32)
            rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
            out_audio = signal.convolve(audio, rir_audio,
                                        mode='full')[:audio_len]
            out_audio = torch.from_numpy(out_audio)
            out_audio = torch.unsqueeze(out_audio, 0)
            sample['wav'] = out_audio
        yield sample


def add_noise(data, noise_source, aug_prob):
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
            key, noise_data = noise_source.random_one()
            if key.startswith('noise'):
                snr_range = [0, 15]
            elif key.startswith('speech'):
                snr_range = [5, 30]
            elif key.startswith('music'):
                snr_range = [5, 15]
            else:
                snr_range = [0, 15]
            _, noise_audio = wavfile.read(io.BytesIO(noise_data))
            noise_audio = noise_audio.astype(np.float32)
            if noise_audio.shape[0] > audio_len:
                start = random.randint(0, noise_audio.shape[0] - audio_len)
                noise_audio = noise_audio[start:start + audio_len]
            else:
                # Resize will repeat copy
                noise_audio = np.resize(noise_audio, (audio_len, ))
            noise_snr = random.uniform(snr_range[0], snr_range[1])
            noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
            noise_audio = np.sqrt(10**(
                (audio_db - noise_db - noise_snr) / 10)) * noise_audio
            out_audio = audio + noise_audio
            out_audio = torch.from_numpy(out_audio)
            out_audio = torch.unsqueeze(out_audio, 0)
            sample['wav'] = out_audio
        yield sample
