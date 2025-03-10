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

import torch
from torch.nn.utils import clip_grad_norm_



class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, data_loader, device, writer, args):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        min_duration = args.get('min_duration', 0)

        num_total_batch = 0
        total_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            key, noisy, target, noisy_lengths, clean_lengths = batch
            noisy = noisy.to(device)
            target = target.to(device)
            noisy_lengths = noisy_lengths.to(device)
            clean_lengths = clean_lengths.to(device)
            num_utts = noisy_lengths.size(0)
            if num_utts == 0:
                continue

            # 모델 Forward Pass
            enhanced_signal = model(noisy)

            # 손실(loss) 계산
            loss = model.module.loss(enhanced_signal, target)

            # 역전파(Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), clip)

            # 그래디언트가 유효할 때만 step 수행
            if torch.isfinite(grad_norm):
                optimizer.step()

            # 총 loss 저장
            total_loss += loss.item()
            num_total_batch += 1

            if batch_idx % log_interval == 0:
                logging.debug(
                    'TRAIN Batch {}/{} loss {:.8f}'.format(
                        epoch, batch_idx, loss.item()))

        # 평균 loss 반환 (에포크 전체에 대한)
        avg_loss = total_loss / num_total_batch if num_total_batch > 0 else 0
        return avg_loss  


    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, noisy, target, noisy_lengths, clean_lengths = batch
                noisy = noisy.to(device)
                target = target.to(device)
                noisy_lengths = noisy_lengths.to(device)
                clean_lengths = clean_lengths.to(device)
                num_utts = noisy_lengths.size(0)
                if num_utts == 0:
                    continue
                enhanced_signal = model(noisy)
                loss = model.module.loss(enhanced_signal, target)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Batch {}/{} loss {:.8f} history loss {:.8f}'
                        .format(epoch, batch_idx, loss.item(),
                                total_loss / num_seen_utts))
        return total_loss / num_seen_utts

    def test(self, model, data_loader, device, args):
        return self.cv(model, data_loader, device, args)
