#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

stage=1
stop_stage=1
num_keywords=11

gpus="0,1,2,3"

kws_dir=../s0/exp/ds_tcn
se_dir=../se0/exp/dccrn

kws_config=$kws_dir/config.yaml
se_config=$se_dir/config.yaml

kws_checkpoint=$kws_dir/avg_10.pt
se_checkpoint=$se_dir/10best_avg.pt

num_average=10

# your data dir
indomain_noisy_dir=../se0/data/noisy_test
noisy_dir=../s0/data/

result_dir=exp/se_dccrn_kws_ds_tcn

. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1 : Evaluate for test noisy dataset ..."
  # Testing
  # for x in noisy_test noise_noisy_test music_noisy_test speech_noisy_test; do
  mkdir -p $result_dir

  for x in noise_noisy_test music_noisy_test speech_noisy_test noisy_test;
  do
    test_dir=$result_dir/${x}
    mkdir -p $test_dir
    python monet/bin/test.py \
      --gpu 0 \
      --config_se $se_config \
      --config_kws $kws_config \
      --checkpoint_se $se_checkpoint \
      --checkpoint_kws $kws_checkpoint \
      --test_data ${noisy_dir}/${x}/data.list \
      --output_dir $test_dir \
      --batch_size 1 \
      --num_workers 8 \
      --num_keywords $num_keywords
  done
fi