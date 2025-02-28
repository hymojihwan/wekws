import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

class LogMelFeatureExtractor(nn.Module):
    """Log Mel-Spectrogram Feature Extractor with Padding Fix"""
    def __init__(self, sample_rate=16000, hidden_dim=256, n_fft=512, hop_length=160, min_length=400):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_length = min_length  # 🔥 최소 길이 보장

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=self.hidden_dim,  
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )

    def forward(self, waveform):
        """
        Args:
            waveform: (B, C, T) or (B, T)
        Returns:
            log_mel_spec: (B, 1, hidden_dim, T)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T) 형태로 변환

        # 🔥 최소 길이 보장 (Zero-padding 추가)
        batch_size, channels, time_steps = waveform.shape
        if time_steps < self.min_length:
            pad_length = self.min_length - time_steps
            waveform = torch.cat([waveform, torch.zeros(batch_size, channels, pad_length)], dim=-1)

        # Log Mel-Spectrogram 변환
        mel_spec = self.mel_transform(waveform)  # (B, 1, F, T)
        log_mel_spec = torch.log(mel_spec + 1e-6)  # (B, 1, hidden_dim, T)
        return log_mel_spec
        
class SubsamplingBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling_rate = 1


class NoSubsampling(SubsamplingBase):
    """No subsampling in accordance to the 'none' preprocessing
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearSubsampling1(SubsamplingBase):
    """Linear transform the input without subsampling
    """
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.ReLU(),
        )
        self.subsampling_rate = 1
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.out(x)
        x = self.dequant(x)
        return x

    def fuse_modules(self):
        torch.quantization.fuse_modules(self, [['out.0', 'out.1']],
                                        inplace=True)


class Conv1dSubsampling1(SubsamplingBase):
    """Conv1d transform without subsampling
    """
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3),
            torch.nn.BatchNorm1d(odim),
            torch.nn.ReLU(),
        )
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x)
        return x

