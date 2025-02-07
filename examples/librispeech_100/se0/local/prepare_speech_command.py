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
    with open(args.wav_list) as f:
        for line in f.readlines():
            # /DB/music_test-clean/237/134493/237-134493-0000.wav
            id_1, id_2, file_name = line.strip().split('/')[-3:]   # 237, 134493, 237-134493-0000.wav
            DB_dir = '/'.join(line.strip().split('/')[:-4])      # /DB
            file_name_new = file_name.split('.')[0]             # 237-134493-0000
            clean_file_name = file_name_new + '.flac'
            wav_id = '_'.join([id_1, id_2, file_name_new])         # 237_134493_237-134493-0000
            
            clean_file_path = os.path.join(DB_dir, "LibriSpeech", args.clean_name, id_1, id_2, clean_file_name)
            noisy_file_path = line.strip()
            
            f_noisy_wav_scp.writelines(wav_id + ' ' + noisy_file_path + '\n')
            f_clean_wav_scp.writelines(wav_id + ' ' + clean_file_path + '\n')
    f_noisy_wav_scp.close()
    f_clean_wav_scp.close()
