import sys
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import ot
import torchaudio.compliance.kaldi as kaldi
import torch.nn as nn
from monet.model.se_model import DCCRN as init_se_model
from monet.model.kws_model import init_model as init_kws_model
from monet.model.loss import criterion

class DifferentiableFeatureExtractor(nn.Module):
    def __init__(self):
        super(DifferentiableFeatureExtractor, self).__init__()

        # 🚀 첫 번째 Convolution (stride=4 → Feature 길이 1/4 축소)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        # 🚀 두 번째 Convolution (stride=4 → Feature 길이 1/16 축소)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # 🚀 세 번째 Convolution (stride=2 → Feature 길이 1/32 축소)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=80, kernel_size=3, stride=4, padding=1)
        self.bn3 = nn.BatchNorm1d(80)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        x = self.conv1(x)  # (batch, 32, time/4)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # (batch, 64, time/16)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)  # (batch, 80, time/32)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)  # (batch, time, feature_dim)
        return x, x.size(1)

class SEKWSModel(nn.Module):
    def __init__(
        self,
        se_configs,
        kws_configs,
    ):
        super().__init__()
        self.se_configs = se_configs
        self.se_model = init_se_model(se_configs)

        self.kws_configs = kws_configs
        self.kws_model = init_kws_model(kws_configs)
        self.kws_loss_type = kws_configs.get('criterion', 'max_pooling')

        self.feature_extracter = DifferentiableFeatureExtractor()


    # def extract_feature(self, enhanced_audio):
    #     """
    #     enhanced_audio: (batch, time) 형태의 오디오 입력
    #     """
    #     batch_size = enhanced_audio.size(0)
    #     feats_list = []
    #     feats_lengths = []
    #     for i in range(batch_size):
    #         waveform = enhanced_audio[i]
    #         if waveform.dim() == 1:
    #             waveform = waveform.unsqueeze(0)
    #         waveform = waveform * (1 << 15)
    #         feats = kaldi.mfcc(
    #             waveform,
    #             num_ceps=80,
    #             num_mel_bins=80,
    #             frame_length=25,
    #             frame_shift=10,
    #             dither=0.0,
    #             energy_floor=0.0,
    #             sample_frequency=16000,
    #         )
    #         feats_tensor = feats.clone().detach()
    #         feats_list.append(feats_tensor)
    #         feats_lengths.append(feats_tensor.size(0))  # 각 샘플의 프레임 수 저장

    #     # 패딩하여 배치로 변환
    #     feats_padded = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True)  # (batch, time, features)
        
    #     # 시퀀스 길이를 텐서로 변환
    #     feats_lengths_tensor = torch.tensor(feats_lengths, dtype=torch.long)  # (batch,)

    #     return feats_padded, feats_lengths_tensor

    def compute_cosine_cost_matrix(self, noisy_features, enhanced_features):
        """
        Cosine Similarity 기반 OT Cost Matrix 계산
        
        Args:
            noisy_features (Tensor): (batch, time, feature_dim)
            enhanced_features (Tensor): (batch, time, feature_dim)

        Returns:
            cost_matrix (Tensor): (batch, time, time)
        """
        noisy_norm = F.normalize(noisy_features, p=2, dim=-1)
        enhanced_norm = F.normalize(enhanced_features, p=2, dim=-1)

        cosine_similarity = torch.matmul(noisy_norm, enhanced_norm.transpose(1, 2))  # (batch, time, time)
        cost_matrix = 1 - cosine_similarity  # Cosine Distance
        return cost_matrix

    def sinkhorn_knopp(self, cost_matrix, epsilon=1.0, max_iter=3):
        """
        Sinkhorn-Knopp Algorithm을 활용한 Optimal Transport 계산 (Batch 지원)
        
        Args:
            cost_matrix (Tensor): (batch, time, time) - Noisy와 Enhanced 간 Cost Matrix
            epsilon (float): Regularization Parameter
            max_iter (int): Sinkhorn Iteration 수

        Returns:
            transport_plan (Tensor): (batch, time, time) - Optimal Transport Matrix
        """
        batch_size, n, m = cost_matrix.shape

        # 초기화
        u = torch.ones(batch_size, n, device=cost_matrix.device) 
        v = torch.ones(batch_size, m, device=cost_matrix.device) 

        # Kernel Matrix 계산
        K = torch.exp(-cost_matrix / epsilon)  # (batch, time, time)

        for _ in range(max_iter):
            u = 1.0 / (torch.matmul(K, v.unsqueeze(-1))).squeeze(-1)  # (batch, time)
            v = 1.0 / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1))).squeeze(-1)  # (batch, time)

        # Transport Plan 계산 (Batch-wise)
        transport_plan = torch.einsum('bti,bi,bj->btj', K, u, v)  # (batch, time, time)

        return transport_plan

    def _calc_wasserstein_loss(self, noisy_features, enhanced_features, epsilon=10.0, max_iter=3):
        """
        Noisy Signal과 Enhanced Signal 간 OT 기반 Wasserstein Loss 계산
        
        Args:
            noisy_features (Tensor): (batch, time, feature_dim)
            enhanced_features (Tensor): (batch, time, feature_dim)
            epsilon (float): Sinkhorn Regularization Parameter
            max_iter (int): Sinkhorn Iteration 수

        Returns:
            wasserstein_loss (Tensor): 최적화된 Wasserstein Loss
            aligned_features (Tensor): OT 기반 정렬된 Feature
        """
        batch_size, time_steps, feature_dim = noisy_features.shape

        # 🚀 배치 단위로 Cost Matrix 계산 (for-loop 제거)
        cost_matrix = self.compute_cosine_cost_matrix(noisy_features, enhanced_features)  # (batch, time, time)

        # 🚀 Sinkhorn Algorithm을 활용한 Transport Plan 계산 (batch 단위)
        transport_plan = self.sinkhorn_knopp(cost_matrix, epsilon, max_iter)  # (batch, time, time)

        # 🚀 정렬된 Enhanced Feature 계산 (batch 단위)
        # aligned_features = torch.einsum('btu,btd->btd', transport_plan, enhanced_features)  # (batch, time, feature_dim)

        # 🚀 Wasserstein 손실 계산 (OT Cost + Entropy Regularization)
        entropy_term = torch.sum(transport_plan * torch.log(transport_plan + 1e-9), dim=(1, 2))
        wasserstein_loss = torch.sum(transport_plan * cost_matrix, dim=(1, 2)) - entropy_term
        wasserstein_loss = wasserstein_loss.mean()  # 배치 평균

        wasserstein_loss = torch.log(1 + wasserstein_loss) # log scaling
        return wasserstein_loss

    def forward(self, noisy_audio, clean_audio, target):
        """
        noisy_audio: (batch, 1(channel), time)
        clean_audio: (batch, 1(channel), time)
        target: (batch, )
        """

        # 1. Speech enhancement
        _, enhanced_audio = self.se_model(noisy_audio)

        # 2. Extract features of enhanced audio
        noisy_feats, noisy_feats_lengths = self.feature_extracter(noisy_audio)
        enhanced_feats, enhanced_feats_lengths = self.feature_extracter(enhanced_audio)
        # clean_feats, clean_feats_lengths = self.feature_extracter(clean_audio)
        # 2-1. OT calculate
        ot_loss_noisy_enhanced = self._calc_wasserstein_loss(noisy_feats, enhanced_feats)
        # ot_loss_enhanced_clean = self._calc_wasserstein_loss(enhanced_feats, clean_feats)
        # ot_loss = ot_loss_noisy_enhanced + ot_loss_enhanced_clean
        ot_loss = ot_loss_noisy_enhanced

        # 3. Keyword spotting
        logits, _ = self.kws_model(enhanced_feats)

        # 4. Compute loss
        loss, se_loss, kws_loss = self._calc_loss_(enhanced_audio, clean_audio, logits, target, noisy_feats_lengths)

        loss = 0.3 * ot_loss + 0.7 * loss
        return loss, logits, se_loss, kws_loss, ot_loss

    
    def _calc_loss_(self, enhanced_audio, clean_audio, logits, target, feats_lengths, alpha=0.7):
        """
        enhanced_audio: Denoised audio from SE model
        clean_audio: Ground truth clean audio
        logits: keyword spotting model output
        target: True labels for KWS
        alpha: Weight for multi-task learning loss
        """

        se_loss = self.se_model.loss(enhanced_audio, clean_audio)
        kws_loss, acc = criterion(self.kws_loss_type, logits, target, feats_lengths,
                                    target_lengths=target.size(-1),
                                    min_duration=0,
                                    validation=False)
        
        loss = (1 - alpha) * se_loss + alpha * kws_loss

        return loss, se_loss, kws_loss

