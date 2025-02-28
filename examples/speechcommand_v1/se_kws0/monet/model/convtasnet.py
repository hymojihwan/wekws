import torch
import torch.nn as nn
import torch.nn.functional as F

from asteroid.models import ConvTasNet

class SEModel(nn.Module):
    def __init__(self, configs):
        super(SEModel, self).__init__()
        self.num_sources = configs['num_sources']
        self.num_blocks = configs['num_blocks']
        self.num_repeats = configs['num_repeats']
        self.loss_mode = configs['loss_mode']
        self.model = ConvTasNet(n_src=self.num_sources, n_blocks=self.num_blocks, n_repeats=self.num_repeats)

    def forward(self, noisy_waveform):
        clean_waveform = self.model(noisy_waveform)  # (batch, 1, time)
        return clean_waveform.squeeze(1)  # (batch, time)


    def loss(self, inputs, labels):
        inputs = inputs.unsqueeze(1)
        if self.loss_mode == 'MSE':
            b, d, t = inputs.shape 
            labels[:,0,:]=0
            labels[:,d//2,:]=0
            return F.mse_loss(inputs, labels, reduction='mean')*d

        elif self.loss_mode == 'MSLE':
            b, d, t = inputs.shape
            labels[:,0,:]=0
            labels[:,d//2,:]=0
            return torch.log(F.mse_loss(inputs, labels, reduction='mean')*d)

        elif self.loss_mode == 'SI-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            # return -(si_snr(inputs, labels))
            lengths = torch.tensor([x.size(-1) for x in labels], dtype=torch.int32).to(inputs.device)
            return -(si_snr_with_mask(inputs, labels, lengths))

        elif self.loss_mode == 'MAE':
            gth_spec, gth_phase = self.stft(labels) 
            b,d,t = inputs.shape 
            return torch.mean(torch.abs(inputs-gth_spec))*d

        elif self.loss_mode == 'SI-SDR':
            return si_sdr_loss(inputs, labels).mean()

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

# class Conv1DBlock(nn.Module):
#     """
#     1D Convolutional block used in the encoder, decoder, and separation network.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
#         super(Conv1DBlock, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = nn.PReLU() if activation else None

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.activation:
#             x = self.activation(x)
#         return x


# class ConvTasNet(nn.Module):
#     def __init__(
#         self, 
#         enc_kernel_size=16, 
#         enc_num_feats=512, 
#         num_blocks=8, 
#         num_layers=4, 
#         bottleneck_channels=128, 
#         num_sources=2
#     ):
#         """
#         Conv-TasNet Model
#         Args:
#             enc_kernel_size: Kernel size of the encoder.
#             enc_num_feats: Number of features in the encoder output.
#             num_blocks: Number of blocks in the temporal convolution network (TCN).
#             num_layers: Number of convolutional layers per TCN block.
#             bottleneck_channels: Number of channels in the bottleneck layer.
#             num_sources: Number of sources to separate.
#         """
#         super(ConvTasNet, self).__init__()
#         self.enc_kernel_size = enc_kernel_size
#         self.enc_num_feats = enc_num_feats
#         self.num_sources = num_sources

#         # Encoder
#         self.encoder = nn.Conv1d(1, enc_num_feats, kernel_size=enc_kernel_size, stride=enc_kernel_size // 2, padding=0)

#         # Bottleneck layer
#         self.bottleneck = nn.Conv1d(enc_num_feats, bottleneck_channels, kernel_size=1)

#         # Temporal Convolution Network (TCN)
#         self.tcn = nn.ModuleList()
#         for _ in range(num_blocks):
#             self.tcn.append(
#                 TCNBlock(
#                     in_channels=bottleneck_channels,
#                     out_channels=bottleneck_channels,
#                     num_layers=num_layers
#                 )
#             )

#         # Mask Generators for each source
#         self.mask_generators = nn.ModuleList([
#             nn.Conv1d(bottleneck_channels, enc_num_feats, kernel_size=1) for _ in range(num_sources)
#         ])

#         # Decoder
#         self.decoder = nn.ConvTranspose1d(
#             enc_num_feats, 1, kernel_size=enc_kernel_size, stride=enc_kernel_size // 2, padding=0
#         )

#     def forward(self, mixture):
#         """
#         Forward pass of the Conv-TasNet model.
#         Args:
#             mixture: Input mixture signal, shape (batch, time).
#         Returns:
#             List of separated signals, each of shape (batch, time).
#         """
#         # Encoder
#         mixture = mixture.unsqueeze(1)  # (batch, 1, time)
#         enc_out = self.encoder(mixture)  # (batch, enc_num_feats, time)

#         # Bottleneck
#         bottleneck_out = self.bottleneck(enc_out)  # (batch, bottleneck_channels, time)

#         # Temporal Convolution Network
#         tcn_out = bottleneck_out
#         for block in self.tcn:
#             tcn_out = block(tcn_out)

#         # Mask generation
#         masks = [F.relu(mask_gen(tcn_out)) for mask_gen in self.mask_generators]  # List of (batch, enc_num_feats, time)

#         # Apply masks and decode
#         separated_signals = []
#         for mask in masks:
#             masked_enc_out = enc_out * mask
#             separated_signal = self.decoder(masked_enc_out)
#             separated_signal = separated_signal.squeeze(1)  # (batch, time)
#             separated_signals.append(separated_signal)

#         return separated_signals


# class TCNBlock(nn.Module):
#     """
#     A single block of Temporal Convolution Network (TCN).
#     """
#     def __init__(self, in_channels, out_channels, num_layers, kernel_size=3, dilation_base=2):
#         super(TCNBlock, self).__init__()
#         self.layers = nn.ModuleList()
#         for i in range(num_layers):
#             dilation = dilation_base**i
#             self.layers.append(
#                 nn.Sequential(
#                     nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation),
#                     nn.BatchNorm1d(out_channels),
#                     nn.PReLU()
#                 )
#             )
#             self.layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))  # Residual connection

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


# if __name__ == "__main__":
#     # Example usage
#     model = ConvTasNet()
#     mixture = torch.rand(4, 16000)  # Batch size = 4, 1-second audio at 16kHz
#     outputs = model(mixture)
#     print(f"Output shape for source 1: {outputs[0].shape}")  # (batch, time)
#     print(f"Output shape for source 2: {outputs[1].shape}")  # (batch, time)