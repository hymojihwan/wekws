#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

stage=3
stop_stage=6

config=conf/dccrn.yaml

gpus="0,1,2,3"

checkpoint=
dir=exp/dccrn

num_average=10
score_checkpoint=$dir/avg_${num_average}

# your data dir
clean_dir=/DB/LibriSpeech/train-clean-100
noise_dir=/home/user/Workspace/MS-SNSD/noise_train
musan_noise_dir=/DB/musan

. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0 : Create mixed audio file for LibriSpeech train-clean-100"
  python local/create_mixed_audio_file_for_train.py \
      --clean_dir /DB/LibriSpeech/train-clean-100 \
      --noise_dir /home/user/Workspace/MS-SNSD/noise_train \
      --noisy_dir /DB/noisy_train-clean-100      

  echo "Stage 0-1 : Create mixed audio file for LibriSpeech dev-clean"
  python local/create_mixed_audio_file_for_train.py \
      --clean_dir /DB/LibriSpeech/dev-clean \
      --noise_dir /home/user/Workspace/MS-SNSD/noise_test \
      --noisy_dir /DB/noisy_dev-clean      
  
  echo "Stage 0-2 : Create mixed audio file for LibriSpeech test-clean"
  python local/create_mixed_audio_file_for_train.py \
      --clean_dir /DB/LibriSpeech/test-clean \
      --noise_dir /home/user/Workspace/MS-SNSD/noise_test \
      --noisy_dir /DB/noisy_test-clean      

  echo "Stage 0-3 : Create mixed outdomain audio file for LibriSpeech test-clean"
  for x in music noise speech;
  do
    python local/create_mixed_audio_file_for_test.py \
        --clean_dir /DB/LibriSpeech/test-clean \
        --noise_dir /DB/musan/${x} \
        --noisy_dir /DB/${x}_test-clean      
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1 : Start preparing Kaldi format files for noisy datasets training & test"
  for x in train-clean-100 dev-clean test-clean;
    do
    data=data/noisy_${x}
    mkdir -p $data
      # make wav.scp utt2spk text file
    find /DB/noisy_${x} -name *.wav | grep -v "_background_noise_" > $data/wav.list
    python local/prepare_speech_command.py --wav_list=$data/wav.list --clean_name=$x --data_dir=$data
    done
  echo "Stage 1-1 : Start preparing Kaldi format files for out-domain noisy testsets"
  for x in music noise speech;
    do
    data=data/${x}_test-clean
    mkdir -p $data
    find /DB/${x}_test-clean   -name *.wav | grep -v "_background_noise_" > $data/wav.list
    python local/prepare_speech_command.py --wav_list=$data/wav.list --clean_name=test_clean --data_dir=$data
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2 : Format datasets"
  for x in noisy_train-clean-100 noisy_dev-clean noisy_test-clean music_test-clean noise_test-clean speech_test-clean; do
    tools/wav_to_duration.sh --nj 8 data/$x/noisy_wav.scp data/$x/noisy_wav.dur
    tools/make_list_se.py data/$x/noisy_wav.scp data/$x/clean_wav.scp \
      data/$x/noisy_wav.dur data/$x/data.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3 : Start training ..."
  mkdir -p $dir
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
   monet/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/noisy_train-clean-100/data.list \
    --cv_data data/noisy_dev-clean/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --min_duration 50 \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4 : Averaging model for extracting 10best ..."
  # Do model average
  python monet/bin/average_model.py --model_dir $dir
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5 : Evaluate for test noisy dataset ..."
  # Testing
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  for x in noisy_test-clean music_test-clean noise_test-clean speech_test-clean;
  do
    output_dir=$result_dir/${x}
    mkdir -p $output_dir
    python monet/bin/test.py \
      --config $dir/config.yaml \
      --checkpoint $dir/10best_avg.pt \
      --test_data data/${x}/data.list \
      --output_dir $output_dir
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6 : Evaluate for test noisy dataset ..."
  # Testing
  # for x in noisy_test noise_noisy_test music_noisy_test speech_noisy_test; do
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir

  python monet/bin/test.py \
    --config $dir/config.yaml \
    --checkpoint $dir/10best_avg.pt \
    --test_data data/speech_noisy_test/data.list \
    --output_dir $result_dir
fi
