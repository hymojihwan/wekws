#!/bin/bash

. ./path.sh

stage=2
stop_stage=4
num_keywords=11

config=conf/convtas_mdtc.yaml

gpus="0,1,2,3"

checkpoint=
dir=exp/convtas_mdtc

num_average=10
score_checkpoint=$dir/10best_avg.pt

# your data dir
download_dir=/home/user/Workspace/wekws/examples/speechcommand_v1/s0/data/local
speech_command_dir=$download_dir/speech_commands_v1

. tools/parse_options.sh || exit 1;

set -euo pipefail


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1 : Start preparing Kaldi format files for noisy datasets training"
    for x in noisy_train noisy_valid noisy_test music_noisy_test speech_noisy_test noise_noisy_test;
    do
        data=data/${x}
        mkdir -p $data
        find $speech_command_dir/${x} -name '*.wav' | grep -v "_background_noise_" > $data/wav.list
        python local/prepare_speech_command.py --wav_list=$data/wav.list --clean_name=$x --data_dir=$data
        tools/wav_to_duration.sh --nj 8 $data/noisy_wav.scp $data/noisy_wav.dur
        tools/make_list.py $data/noisy_wav.scp $data/clean_wav.scp $data/text \
        $data/noisy_wav.dur $data/data.list
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2 : Start training ..."
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
  echo "Stage 3 : Averaging model"
  python monet/bin/average_model.py --model_dir $dir
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Testing
  echo "Stage 4 : Compute Accuracy"
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python monet/bin/test.py --gpu 3 \
    --config $dir/config.yaml \
    --test_data data/speech_noisy_test/data.list \
    --batch_size 256 \
    --num_workers 8 \
    --checkpoint $score_checkpoint \
    --output_dir $result_dir 
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Testing
  echo "Stage 5 : Compute Accuracy"
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python monet/bin/new_test.py \
    --config $dir/config.yaml \
    --test_data data/speech_noisy_test/data.list \
    --checkpoint $score_checkpoint \
    --output_dir $result_dir 
fi

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   # Testing
#   echo "Stage 4 : Compute Accuracy"
#   result_dir=$dir/test_$(basename $score_checkpoint)
#   mkdir -p $result_dir
#   python monet/bin/compute_accuracy.py --gpu 3 \
#     --config $dir/config.yaml \
#     --test_data data/speech_noisy_test/data.list \
#     --batch_size 256 \
#     --num_workers 8 \
#     --checkpoint $score_checkpoint \
#     --output_dir $result_dir 
# fi
