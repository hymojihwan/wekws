#!/usr/bin/env python3
# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

CLASSES = 'unknown, yes, no, up, down, left, right, on, off, stop, go'.split(
    ', ')
CLASS_TO_IDX = {CLASSES[i]: str(i) for i in range(len(CLASSES))}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepare kaldi format file for google speech command')
    parser.add_argument(
        '--wav_list',
        required=True,
        help='full path of a wav file in google speech command dataset')
    parser.add_argument(
        '--clean_name',
        required=True,
        help='noise name')
            
    parser.add_argument('--data_dir',
                        required=True,
                        help='folder to write kaldi format files')
    args = parser.parse_args()

    data_dir = args.data_dir
    f_noisy_wav_scp = open(os.path.join(data_dir, 'noisy_wav.scp'), 'w')
    f_clean_wav_scp = open(os.path.join(data_dir, 'clean_wav.scp'), 'w')
    f_text = open(os.path.join(data_dir, 'text'), 'w')
    with open(args.wav_list) as f:
        for line in f.readlines():
            # /home/user/Workspace/wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/train/one/cb2929ce_nohash_2.wav
            keyword, file_name = line.strip().split('/')[-2:]   # bird, c2d15ea5_nohash_0.wav
            clean_file_name = line.strip().split('/')[-1]       # c2d15ea5_nohash_0.wav
            DB_dir = '/'.join(line.strip().split('/')[:-3])      # /home/user/Workspace/wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/train
            
            file_name_new = file_name.split('.')[0]             # c2d15ea5_nohash_0
            wav_id = '_'.join([keyword, file_name_new])         # bird_c2d15ea5_nohash_0         
            
            clean_name = args.clean_name.split('_')[-1]
            clean_file_path = os.path.join(DB_dir, clean_name, keyword, clean_file_name)
            noisy_file_path = line.strip()
            
            f_noisy_wav_scp.writelines(wav_id + ' ' + noisy_file_path + '\n')
            f_clean_wav_scp.writelines(wav_id + ' ' + clean_file_path + '\n')

            label = CLASS_TO_IDX[
                keyword] if keyword in CLASS_TO_IDX else CLASS_TO_IDX["unknown"]
            f_text.writelines(wav_id + ' ' + str(label) + '\n')

    f_noisy_wav_scp.close()
    f_clean_wav_scp.close()
    f_text.close()