#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

stage=2
stop_stage=2
num_keywords=11

config=conf/dccrn.yaml

gpus="0"

checkpoint=
dir=exp/dccrn

num_average=10
score_checkpoint=$dir/avg_${num_average}.pt

# your data dir
download_dir=/DB
speech_command_dir=$download_dir/speech_commands_v1
. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Download and extract all datasets"
  local/data_download.sh --dl_dir $download_dir
  python local/split_dataset.py $download_dir/speech_commands_v1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Start preparing Kaldi format files"
  for x in noisy_train noisy_test noisy_valid;
  do
    data=data/$x
    mkdir -p $data
    # make wav.scp utt2spk text file
    find $speech_command_dir/$x -name *.wav | grep -v "_background_noise_" > $data/wav.list
    python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Format datasets"
  for x in noisy_train noisy_valid noisy_test; do
    tools/wav_to_duration.sh --nj 8 data/$x/wav.scp data/$x/wav.dur
    tools/make_list.py data/$x/wav.scp data/$x/text \
      data/$x/wav.dur data/$x/data.list
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start training ..."
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Do model average
  python wekws/bin/average_model.py \
    --dst_model $score_checkpoint \
    --src_path $dir  \
    --num ${num_average} \
    --val_best
  result_dir=$dir/test_$(basenam e $score_checkpoint)
  mkdir -p $result_dir
  # python wekws/bin/score.py \
  #   --config $dir/config.yaml \
  #   --test_data data/test/data.list \
  #   --batch_size 256 \
  #   --checkpoint $score_checkpoint \
  #   --score_file $result_dir/score.txt \
  #   --num_workers 8
  # first_keyword=0
  # last_keyword=$(($num_keywords+$first_keyword-1))
  # for keyword in $(seq $first_keyword $last_keyword); do
  #   python wekws/bin/compute_det.py \
  #     --keyword $keyword \
  #     --test_data data/test/data.list \
  #     --score_file $result_dir/score.txt \
  #     --stats_file $result_dir/stats.${keyword}.txt
  # done
  # Testing
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python wekws/bin/compute_accuracy.py --gpu 1 \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --batch_size 256 \
    --num_workers 8 \
    --checkpoint $score_checkpoint
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
