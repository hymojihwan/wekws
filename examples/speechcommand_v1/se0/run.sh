#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

stage=7
stop_stage=7
num_keywords=11

config=conf/dccrn.yaml

gpus="0,1,2,3"

checkpoint=
dir=exp/dccrn

num_average=10
score_checkpoint=$dir/avg_${num_average}

# your data dir
download_dir=/home/user/Workspace/wekws/examples/speechcommand_v1/s0/data/local
speech_command_dir=$download_dir/speech_commands_v1
noise_dir=/home/user/Workspace/MS-SNSD
musan_noise_dir=/DB/musan

. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ]; then
  echo "Stage -3 : Download and extract all datasets"
  local/data_download.sh --dl_dir $download_dir
  python local/split_dataset.py $download_dir/speech_commands_v1
fi

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  echo "Stage -2 : Create mixed audio file for train & valid dataset"
  for x in train valid;
    do
      noisy_dir=$speech_command_dir/noisy_${x}
      if [ ! -d "$noisy_dir" ]; then
        echo "Creating noisy directory and processing: $noisy_dir"
        python local/create_mixed_audio_file_for_train.py --clean_dir=$speech_command_dir --split_name=$x --noise_dir=$noise_dir/noise_train --noisy_dir=$noisy_dir
      else
        echo "Noisy directory already exists: $noisy_dir. Skipping..."
      fi
    done
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Stage -1 : Start preparing Kaldi format files for noisy datasets training"
  for x in train valid;
    do
    data_combined=data/noisy_${x}
    mkdir -p $data_combined
    > $data_combined/wav.list
      # make wav.scp utt2spk text file
      find $speech_command_dir/noisy_${x} -name *.wav | grep -v "_background_noise_" >> $data_combined/wav.list
      python local/prepare_speech_command.py --wav_list=$data_combined/wav.list --clean_name=$x --data_dir=$data_combined
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0 : Format datasets"
  for x in noisy_train noisy_valid; do
    tools/wav_to_duration.sh --nj 8 data/$x/noisy_wav.scp data/$x/noisy_wav.dur
    tools/make_list_se.py data/$x/noisy_wav.scp data/$x/clean_wav.scp \
      data/$x/noisy_wav.dur data/$x/data.list
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1 : Start training ..."
  mkdir -p $dir
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
   monet/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/noisy_train/data.list \
    --cv_data data/noisy_valid/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2 : Averaging model for extracting 10best ..."
  # Do model average
  python monet/bin/average_model.py --model_dir $dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3 : Evaluate for test noisy dataset ..."
  # Testing
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python monet/bin/test.py \
    --config $dir/config.yaml \
    --checkpoint $dir/10best_avg.pt \
    --test_data data/test/data.list \
    --output_dir $result_dir
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  python wekws/bin/export_jit.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --jit_model $dir/$jit_model

  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5 : Create mixed audio file for test dataset"
  for x in test;
    do
      noisy_dir=$speech_command_dir/noisy_${x}
      if [ ! -d "$noisy_dir" ]; then
        echo "Creating noisy directory and processing: $noisy_dir"
        # For MS-SNSD test noise combine ( indomain noise test)
        echo "Creating in-domain noisy test datasets (MS-SNSD)"
        python local/create_mixed_audio_file_for_train.py --clean_dir=$speech_command_dir/$x --noise_dir=$noise_dir/noise_test --noisy_dir=$noisy_dir
      else
        echo "Noisy directory already exists: $noisy_dir. Skipping..."
      fi
      out_dir=$speech_command_dir/noise_noisy_${x}
      if [ ! -d "$out_dir" ]; then
       # For MUSAN test noise combine ( outdomain noise test )
        echo "Creating out-domain noisy test datasets (MUSAN)"
        python local/create_mixed_audio_file_for_test.py --clean_dir=$speech_command_dir/$x --noise_dir=$musan_noise_dir --noisy_dir=data
      else
        echo "Noisy directory already exists: $out_dir. Skipping..."
      fi

    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6 : Format noisy test datasets"
  for x in noisy_test noise_noisy_test music_noisy_test speech_noisy_test; do
    noisy_data_dir=$speech_command_dir/$x
    data=data/$x
    mkdir -p $data
    > $data/wav.list
      # make wav.scp utt2spk text file
      find $noisy_data_dir -name *.wav | grep -v "_background_noise_" >> $data/wav.list
      python local/prepare_speech_command.py --wav_list=$data/wav.list --clean_name="test" --data_dir=$data

    tools/wav_to_duration.sh --nj 8 $data/noisy_wav.scp $data/noisy_wav.dur
    tools/make_list_se.py $data/noisy_wav.scp $data//clean_wav.scp \
      $data/noisy_wav.dur $data/data.list
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7 : Evaluate for test noisy dataset ..."
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