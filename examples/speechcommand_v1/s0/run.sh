#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

stage=6
stop_stage=6
num_keywords=11

config=conf/mdtc.yaml
norm_mean=false
norm_var=false
gpus="0,1,2,3"

enhanced_model=convtas
checkpoint=
dir=exp/mdtc

num_average=10
score_checkpoint=$dir/avg_${num_average}.pt

# your data dir
download_dir=./data/local
speech_command_dir=$download_dir/speech_commands_v1
noisy_dir=/DB/speech_commands_v1
. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Stage -1 : Download and extract all datasets"
  local/data_download.sh --dl_dir $download_dir
  python local/split_dataset.py $download_dir/speech_commands_v1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0 : Start preparing Kaldi format files"
  for x in train test valid;
  do
    data=data/$x
    mkdir -p $data
    # make wav.scp utt2spk text file
    find $speech_command_dir/$x -name *.wav | grep -v "_background_noise_" > $data/wav.list
    python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1 : Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp data/train/wav.scp \
    --out_cmvn data/train/global_cmvn

  for x in train valid test; do
    tools/wav_to_duration.sh --nj 8 data/$x/wav.scp data/$x/wav.dur
    tools/make_list.py data/$x/wav.scp data/$x/text \
      data/$x/wav.dur data/$x/data.list
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2 : Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
   wekws/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/train/data.list \
    --cv_data data/valid/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    $cmvn_opts \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Do model average
  # echo "Stage 3 : Averaging model"
  # python wekws/bin/average_model.py \
  #   --dst_model $score_checkpoint \
  #   --src_path $dir  \
  #   --num ${num_average} \
  #   --val_best

  # Testing
  echo "Stage 3 : Compute Accuracy"
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python wekws/bin/compute_accuracy.py --gpu 3 \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --batch_size 256 \
    --num_workers 8 \
    --checkpoint $score_checkpoint
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4 : Export jit & onnx"
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
    echo "Stage 5 : Prepare noisy datasets"
    for x in noisy_test noisy_valid;
      do
      data_combined=data/$x
      mkdir -p $data_combined
      > $data_combined/wav.list
        # make wav.scp utt2spk text file
        for noise in speech music noise;
          do
            data=data/${noise}_$x
            mkdir -p $data
            find $noisy_dir/$x/$noise -name *.wav | grep -v "_background_noise_" > $data/wav.list
            find $noisy_dir/$x/$noise -name *.wav | grep -v "_background_noise_" >> $data_combined/wav.list
            python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
          done
      python local/prepare_speech_command.py --wav_list=$data_combined/wav.list --data_dir=$data_combined
      done

    for x in noisy_valid noisy_test; do
      for noise in speech music noise; do
          data=data/${noise}_${x}
          tools/wav_to_duration.sh --nj 8 $data/wav.scp $data/wav.dur
          tools/make_list.py $data/wav.scp $data/text \
              $data/wav.dur $data/data.list
      done

      # 합쳐진 데이터셋 처리
      data_combined=data/${x}
      tools/wav_to_duration.sh --nj 8 $data_combined/wav.scp $data_combined/wav.dur
      tools/make_list.py $data_combined/wav.scp $data_combined/text \
          $data_combined/wav.dur $data_combined/data.list
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6 : Compute Accuracy of Noisy datasets"
    result_dir=$dir/test_$(basename $score_checkpoint)
    mkdir -p $result_dir
    python monet/bin/compute_accuracy.py --gpu 0 \
      --config $dir/config.yaml \
      --test_data \
        data/test/data.list \
        data/convtasnet_enhanced_noisy_test/data.list \
        data/convtasnet_enhanced_noise_noisy_test/data.list \
        data/convtasnet_enhanced_music_noisy_test/data.list \
        data/convtasnet_enhanced_speech_noisy_test/data.list \
      --batch_size 256 \
      --num_workers 8 \
      --checkpoint $score_checkpoint \
      --result_dir $result_dir
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7 : Prepare enhanced noisy datasets"
    for x in enhanced_noisy_test enhanced_music_noisy_test enhanced_noise_noisy_test enhanced_speech_noisy_test;
    do
      data=data/${enhanced_model}_${x}
      mkdir -p $data
      find $speech_command_dir/$enhanced_model/$x -name '*.wav' | grep -v "_background_noise_" > $data/wav.list
      python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
      tools/wav_to_duration.sh --nj 8 $data/wav.scp $data/wav.dur
      tools/make_list.py $data/wav.scp $data/text \
              $data/wav.dur $data/data.list
    done

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8 : Compute Accuracy of Enhanced Noisy datasets"
    for x in enhanced_noisy_test enhanced_music_noisy_test enhanced_noise_noisy_test enhanced_speech_noisy_test;
    do
    result_dir=$dir/test_$(basename $score_checkpoint)
    mkdir -p $result_dir
    python monet/bin/compute_accuracy.py --gpu 0 \
      --config $dir/config.yaml \
      --test_data \
        data/enhanced_noisy_test/data.list \
        data/enhanced_noise_noisy_test/data.list \
        data/enhanced_music_noisy_test/data.list \
        data/enhanced_speech_noisy_test/data.list \
      --batch_size 256 \
      --num_workers 8 \
      --checkpoint $score_checkpoint \
      --result_dir $result_dir
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Stage 9 : Prepare enhanced noisy training datasets"
    for x in enhanced_noisy_train;
    do
      data=data/${enhanced_model}_$x
      mkdir -p $data
      find $speech_command_dir/${enhanced_model}/$x -name '*.wav' | grep -v "_background_noise_" > $data/wav.list
      python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
      tools/wav_to_duration.sh --nj 8 $data/wav.scp $data/wav.dur
      tools/make_list.py $data/wav.scp $data/text \
              $data/wav.dur $data/data.list
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Stage 10 : Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
   wekws/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/${enhanced_model}_enhanced_noisy_train/data.list \
    --cv_data data/noisy_valid/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    $cmvn_opts \
    ${checkpoint:+--checkpoint $checkpoint}
fi
