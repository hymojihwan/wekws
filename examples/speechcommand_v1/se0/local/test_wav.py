import argparse
import os
import yaml
import librosa
import soundfile
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from monet.model.se_model import init_model
from monet.utils.checkpoint import load_checkpoint

FIG_SIZE = (15, 10)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def plot_spec(enhanced_spec, noisy_spec, out_dir):
    # plt.figure(figsize=FIG_SIZE)
    # librosa.display.specshow(noisy_spec.numpy(), hop_length=100)
    # plt.subplot(211)    
    # plt.plot(noisy_spec.numpy())
    librosa.display.specshow(enhanced_spec.numpy(), hop_length=100)
    plt.colorbar(format="%+2.0f dB")
    # plt.title('noisy')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    # librosa.display.specshow(enhanced_spec.numpy(), hop_length=100)
    # plt.subplot(212)    
    # plt.plot(enhanced_spec.numpy())
    plt.title('enhanced')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    save_path = os.path.join(out_dir, 'spec.png')
    plt.savefig(save_path)

def save_wav(enhanced_wav, noisy_wav, out_dir):
    enhanced_save_wav = os.path.join(out_dir, 'enhanced.wav')
    noisy_save_wav = os.path.join(out_dir, 'noisy.wav')
    
    torchaudio.save(enhanced_save_wav, enhanced_wav, 16000)
    torchaudio.save(noisy_save_wav, noisy_wav, 16000)

def se(model, noisy_path):
    ''' Cross validation on
    '''
    model.eval()
    
    with torch.no_grad():
        
        w, _ = torchaudio.load(noisy_path)

        hop_length = 100
        n_fft = 512
        noisy_stft = librosa.stft(np.array(w), n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(noisy_stft)
        log_spec = librosa.amplitude_to_db(mag)
        
        enhanced_spec, enhanced_wav = model(w)
        enhanced_stft = librosa.stft(np.array(enhanced_wav), n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(enhanced_stft)
        en_spec = librosa.amplitude_to_db(mag)
            
    return torch.Tensor(en_spec).squeeze(), torch.Tensor(log_spec).squeeze(), enhanced_wav, w

if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs['model'])
    infos = load_checkpoint(model, args.checkpoint)

    enhanced_spec, noisy_spec, enhanced_wav, noisy_wav = se(model, args.noisy_file)

    plot_spec(enhanced_spec, noisy_spec, args.out_dir)
    save_wav(enhanced_wav, noisy_wav, args.out_dir)






