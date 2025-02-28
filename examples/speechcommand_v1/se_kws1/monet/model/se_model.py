import torch
import torch.nn as nn
import os
import sys

import torch.nn.functional as F
from torchmetrics import MeanSquaredLogError
from monet.model.show import show_params, show_model
from monet.model.conv_stft import ConvSTFT, ConviSTFT 
from monet.model.complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm

from asteroid.models import ConvTasNet


class SEModel(nn.Module):
    def __init__(self, num_sources=1, num_blocks=8, num_repeats=3):
        super(SEModel, self).__init__()
        self.model = ConvTasNet(n_src=num_sources, n_blocks=num_blocks, n_repeats=num_repeats)

    def forward(self, noisy_waveform):
        clean_waveform = self.model(noisy_waveform)  # (batch, 1, time)
        return clean_waveform.squeeze(1)  # (batch, time)


class DCCRN(nn.Module):

    def __init__(
                    self, configs
                ):
        ''' 
            
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(DCCRN, self).__init__()

        # for fft 
        self.win_len = configs['win_len']
        self.win_inc = configs['hop_len']
        self.fft_len = configs['n_fft']
        self.win_type = configs['win_type']

        input_dim = self.win_len
        output_dim = self.win_len
        
        self.rnn_units = configs['rnn_units']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = configs['rnn_layers']
        self.kernel_size = configs['kernel_size']
        #self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        #self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2]+configs['kernel_num']
        self.masking_mode = configs['masking_mode']
        self.use_clstm = configs['use_clstm']
        self.bidirectional = configs['bidirectional']
        self.use_clstm = configs['use_clstm']
        self.use_cbn = configs['use_cbn']
        self.loss_mode = configs['loss_mode']
        
        fac = 2 if self.bidirectional else 1 


        fix=True
        self.fix = fix
        self.msle = MeanSquaredLogError()

        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=fix)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    #nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx+1]) if not self.use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len//(2**(len(self.kernel_num))) 

        if self.use_clstm: 
            rnns = []
            for idx in range(self.rnn_layers):
                rnns.append(
                        NavieComplexLSTM(
                        input_size= hidden_dim*self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=self.bidirectional,
                        batch_first=False,
                        projection_dim= hidden_dim*self.kernel_num[-1] if idx == self.rnn_layers-1 else None,
                        )
                    )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                    input_size= hidden_dim*self.kernel_num[-1],
                    hidden_size=self.rnn_units,
                    num_layers=2,
                    dropout=0.0,
                    bidirectional=self.bidirectional,
                    batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim*self.kernel_num[-1])

        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2,0),
                        output_padding=(1,0)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx-1]) if not self.use_cbn else ComplexBatchNorm(self.kernel_num[idx-1]),
                    #nn.ELU()
                    nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2,0),
                        output_padding=(1,0)
                    ),
                    )
                )
        
        show_model(self)
        show_params(self)
        self.flatten_parameters() 

    def flatten_parameters(self): 
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, lens=None):
        specs = self.stft(inputs)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        ''' 

        out = cspecs
        encoder_out = []
        
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
        #    print('encoder', out.size())
            encoder_out.append(out)
        
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:,:,:channels//2]
            i_rnn_in = out[:,:,channels//2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2*dims])
        
            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2, dims]) 
            out = torch.cat([r_rnn_in, i_rnn_in],2)
        
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels*dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])
       
        out = out.permute(1, 2, 3, 0)
        
        for idx in range(len(self.decoder)):
            out = complex_cat([out,encoder_out[-1 - idx]],1)
            out = self.decoder[idx](out)
            out = out[...,1:]
        #    print('decoder', out.size())
        mask_real = out[:,0]
        mask_imag = out[:,1] 
        mask_real = F.pad(mask_real, [0,0,1,0])
        mask_imag = F.pad(mask_imag, [0,0,1,0])
        
        if self.masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2(
                            imag_phase,
                            real_phase
                        ) 

            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags 
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase) 
        elif self.masking_mode == 'C':
            real,imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif self.masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag 
        
        out_spec = torch.cat([real, imag], 1) 
        out_wav = self.istft(out_spec)
         
        out_wav = torch.squeeze(out_wav, 1)
        #out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav,-1,1)
        return out_wav, encoder_out

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

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

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data

def l2_norm(s1, s2):
    """ L2 Norm calculation """
    return torch.sum((s1 - s2) ** 2, dim=-1, keepdim=True)

# def l2_norm(s1, s2):
#     #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
#     #norm = torch.norm(s1*s2, 1, keepdim=True)
#     norm = torch.sum(s1*s2, -1, keepdim=True)
#     return norm 

def si_snr(s1, s2, eps=1e-8):
    s1 = remove_dc(s1)
    s2 = remove_dc(s2)

    s1_s2_norm = torch.sum(s1 * s2, dim=-1, keepdim=True)
    s2_s2_norm = torch.sum(s2 * s2, dim=-1, keepdim=True)
    s_target = (s1_s2_norm / (s2_s2_norm + eps)) * s2
    e_noise = s1 - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    noise_norm = torch.sum(e_noise ** 2, dim=-1, keepdim=True)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

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
