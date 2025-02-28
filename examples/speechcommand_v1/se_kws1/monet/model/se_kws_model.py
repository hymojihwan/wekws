import sys
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as kaldi

import torch.nn as nn

# from monet.model.se_model import DCCRN as init_se_model
from monet.model.se_model import SEModel as init_se_model
from monet.model.kws_model import init_model as init_kws_model

class SEKWSModel(nn.Module):
    def __init__(
        self,
        se_configs,
        kws_configs,
    ):
        super().__init__()
        self.se_configs = se_configs
        # DCCRN
        # self.se_model = init_se_model(se_configs)
        # ConvTasNet
        self.se_model = init_se_model(se_configs['num_sources'],
                                    se_configs['num_blocks'],
                                    se_configs['num_repeats'])
        self.se_loss_type = se_configs.get('loss_mode', 'SI-SNR')

        self.loss_weight = se_configs.get('loss_weight', 0.01)
        self.injection_weight = se_configs.get('injection_weight', 0.1)

        self.kws_configs = kws_configs
        self.kws_model = init_kws_model(kws_configs)
        self.kws_loss_type = kws_configs.get('criterion', 'ce')
        # self.spec_transform = T.MelSpectrogram(
        #     sample_rate=16000,
        #     n_mels=80,
        #     n_fft=512,
        #     hop_length=160
        # )
    def compute_mfcc(self, enhanced_audio):
        
        batch_size = enhanced_audio.size(0)
        enhanced_feats = []
        for b in range(batch_size):

            enhanced_before_feat = enhanced_audio[b] * (1 << 15)  # Kaldi는 16-bit PCM 스케일 사용
            enhanced_feat = kaldi.mfcc(
                enhanced_before_feat,
                num_ceps=80,  # 기존 num_ceps 설정
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000
            )
            enhanced_feats.append(enhanced_feat)
        
        enhanced_feats = torch.stack(enhanced_feats)
        return enhanced_feats

    def forward(self, noisy_audio, clean_audio, target):
        """
        noisy_audio: (batch, 1(channel), time)
        clean_audio: (batch, 1(channel), time)
        target: (batch, )
        """
        # 1. Speech enhancement

        # enhanced_audio, _ = self.se_model(noisy_audio) # enhanced_audio = [B, 64000]
        enhanced_audio = self.se_model(noisy_audio)
        if enhanced_audio.dim() == 2:
            enhanced_audio = enhanced_audio.unsqueeze(1)
        # 2. Audio Injection
        injected_audio = self.injection_weight * enhanced_audio + (1 - self.injection_weight) * noisy_audio
        # 3. Keyword spotting
        # enhanced_spec = self.spec_transform(injected_audio)  # (batch, 80, time)
        enhanced_feats = self.compute_mfcc(enhanced_audio)
        
        if enhanced_feats.dim() == 4 and enhanced_feats.shape[1] == 1:
            enhanced_feats = enhanced_feats.squeeze(1)  # (batch, 80, time)

        if enhanced_feats.dim() == 3 and enhanced_feats.shape[1] == 80:  
            enhanced_feats = enhanced_feats.permute(0, 2, 1)  # (batch, time, 80)
        logits, _ = self.kws_model(enhanced_feats)

        # 4. Compute loss
        se_loss = self._calc_se_loss_(enhanced_audio, clean_audio)
        kws_loss = self._calc_kws_loss_(logits, target)
        loss = self.loss_weight * se_loss + kws_loss

        return loss, logits, se_loss, kws_loss

    def _calc_se_loss_(self, enhanced_audio, clean_audio):
        if enhanced_audio.dim() == 2:
            enhanced_audio = enhanced_audio.unsqueeze(1)
        if self.se_loss_type == 'MSE':
            b, d, t = enhanced_audio.shape 
            clean_audio[:,0,:]=0
            clean_audio[:,d//2,:]=0
            return F.mse_loss(enhanced_audio, clean_audio, reduction='mean')*d

        elif self.se_loss_type == 'MSLE':
            b, d, t = enhanced_audio.shape
            clean_audio[:,0,:]=0
            clean_audio[:,d//2,:]=0
            return torch.log(F.mse_loss(enhanced_audio, clean_audio, reduction='mean')*d)

        elif self.se_loss_type == 'SI-SNR':
            lengths = torch.tensor([x.size(-1) for x in clean_audio], dtype=torch.int32).to(enhanced_audio.device)
            return -(si_snr(enhanced_audio, clean_audio))
            # return -(si_snr_with_mask(enhanced_audio, clean_audio, lengths))

        elif self.se_loss_type == 'MAE':
            gth_spec, gth_phase = self.stft(clean_audio) 
            b,d,t = enhanced_audio.shape 
            return torch.mean(torch.abs(enhanced_audio-gth_spec))*d

        elif self.se_loss_type == 'SI-SDR':
            return si_sdr_loss(enhanced_audio, clean_audio).mean()

    def _calc_kws_loss_(self, logits, target):
        if self.kws_loss_type == 'ce':
            kws_loss = F.cross_entropy(logits, target.type(torch.int64))
            return kws_loss

def si_sdr_loss(est_targets, targets, eps=1e-8):
    """ SI-SDR Loss (Speech Separation 최적) """
    s1_s2_norm = torch.sum(est_targets * targets, dim=-1, keepdim=True)
    s2_s2_norm = torch.sum(targets * targets, dim=-1, keepdim=True)
    s_target = (s1_s2_norm / (s2_s2_norm + eps)) * targets
    e_noise = est_targets - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1)
    noise_norm = torch.sum(e_noise ** 2, dim=-1)

    sdr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return -sdr.mean()

def si_snr(s1, s2, eps=1e-8):
    """
    SI-SNR 계산 공식
    Args:
        s1 (Tensor): 모델 출력 (batch, time)
        s2 (Tensor): 정답 신호 (batch, time)
    Returns:
        Tensor: 평균 SI-SNR 값
    """
    s1 = s1 - s1.mean(dim=-1, keepdim=True)
    s2 = s2 - s2.mean(dim=-1, keepdim=True)

    s1_s2_norm = torch.sum(s1 * s2, dim=-1, keepdim=True)
    s2_s2_norm = torch.sum(s2 * s2, dim=-1, keepdim=True)
    s_target = (s1_s2_norm / (s2_s2_norm + eps)) * s2
    e_noise = s1 - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1)
    noise_norm = torch.sum(e_noise ** 2, dim=-1)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return snr.mean()

def si_snr_with_mask(s1, s2, lengths, eps=1e-8):
    """
    SI-SNR 계산 시 패딩된 부분을 제외하는 함수 (배치 지원).

    Args:
        s1 (Tensor): 모델 출력 (batch, channels, time)
        s2 (Tensor): 참조 신호 (batch, channels, time)
        lengths (Tensor): 실제 신호 길이 (batch,)
        eps (float): 작은 값으로 0 나눔 방지

    Returns:
        Tensor: 평균 SI-SNR 값
    """
    batch_size, channels, max_length = s1.size()

    # 패딩 마스크 생성 (1: 유효한 데이터, 0: 패딩)
    mask = torch.arange(max_length).to(s1.device).unsqueeze(0).unsqueeze(0) < lengths.view(-1, 1, 1)

    # 마스크 적용
    s1_masked = s1 * mask
    s2_masked = s2 * mask

    # SI-SNR 계산
    s1_s2_norm = torch.sum(s1_masked * s2_masked, dim=-1, keepdim=True)  # (batch, channels, 1)
    s2_s2_norm = torch.sum(s2_masked * s2_masked, dim=-1, keepdim=True)  # (batch, channels, 1)
    s_target = (s1_s2_norm / (s2_s2_norm + eps)) * s2_masked
    e_noise = s1_masked - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    noise_norm = torch.sum(e_noise ** 2, dim=-1, keepdim=True)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)

    return torch.mean(snr)