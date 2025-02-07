# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
import argparse
import soundfile as sf
import array
import wave

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True, help="Path to LibriSpeech clean-100 directory")
    parser.add_argument('--noise_dir', type=str, required=True, help="Path to noise_train directory")
    parser.add_argument('--noisy_dir', type=str, required=True, help="Path to save noisy LibriSpeech data")
    args = parser.parse_args()
    return args

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, sample_rate, amp):
    if len(amp) == 0:
        print(f"Warning: Empty waveform, skipping {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)  # 모노
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(array.array('h', amp.astype(np.int16)).tobytes())

if __name__ == '__main__':
    args = get_args()

    # LibriSpeech에서 .flac 파일 찾기
    clean_files = glob.glob(os.path.join(args.clean_dir, "**", "*.flac"), recursive=True)
    noise_files = glob.glob(os.path.join(args.noise_dir, "**", "*.wav"), recursive=True)

    if not noise_files:
        raise ValueError("No noise files found in {}".format(args.noise_dir))

    for clean_file in clean_files:
        # FLAC 파일 로드
        clean_amp, sample_rate = sf.read(clean_file, dtype='int16')
        clean_amp = clean_amp.astype(np.float64)  # float64로 변환하여 연산 안정화
        clean_rms = cal_rms(clean_amp)

        # 랜덤 노이즈 선택
        noise_file = random.choice(noise_files)
        noise_amp, noise_rate = sf.read(noise_file, dtype='int16')
        noise_amp = noise_amp.astype(np.float64)

        # 샘플링 레이트 맞추기 (LibriSpeech는 16kHz)
        if noise_rate != sample_rate:
            noise_amp = sf.resample(noise_amp, sample_rate, noise_rate)

        # 노이즈 길이가 짧으면 반복해서 길이를 맞춤
        if len(noise_amp) < len(clean_amp):
            repeat_factor = int(np.ceil(len(clean_amp) / len(noise_amp)))
            noise_amp = np.tile(noise_amp, repeat_factor)

        # 랜덤한 위치에서 노이즈 선택
        start = random.randint(0, len(noise_amp) - len(clean_amp))
        divided_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = cal_rms(divided_noise_amp)

        # SNR (0~20dB) 적용하여 노이즈 스케일링
        snr = random.uniform(0.0, 20.0)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = clean_amp + adjusted_noise_amp

        # Clipping 방지
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max() > max_int16 or mixed_amp.min() < min_int16:
            reduction_rate = min(max_int16 / mixed_amp.max(), min_int16 / mixed_amp.min())
            mixed_amp *= reduction_rate

        # 저장할 경로 설정
        relative_path = os.path.relpath(clean_file, args.clean_dir)
        noisy_file_path = os.path.join(args.noisy_dir, relative_path).replace(".flac", ".wav")

        save_waveform(noisy_file_path, sample_rate, mixed_amp)

        print(f"Saved: {noisy_file_path}")