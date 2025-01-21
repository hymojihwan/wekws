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
    parser.add_argument('--data_dir',
                        required=True,
                        help='folder to write kaldi format files')
    args = parser.parse_args()

    data_dir = args.data_dir
    f_wav_scp = open(os.path.join(data_dir, 'wav.scp'), 'w')
    f_text = open(os.path.join(data_dir, 'text'), 'w')
    with open(args.wav_list) as f:
        for line in f.readlines():
            keyword, file_name = line.strip().split('/')[-2:]
            clean_file_name = line.strip().split('/')[-1]
            DB_dir = '/'.join(line.strip().split('/')[:3])
            
            file_name_new = file_name.split('.')[0]
            wav_id = '_'.join([keyword, file_name_new])
            train_mode = line.strip().split('/')[3]
            clean_mode = train_mode.split('_')[-1]
            clean_file_dir = os.path.join(DB_dir, clean_mode, keyword, clean_file_name)
            if train_mode == "noisy_train":
                noise_list = ["music", "noise", "speech"]
                random_noise = random.randrange(0,3)
            else:
                noise_list = ["babble", "cafe", "livingroom", "office"]
                random_noise = random.randrange(0,4)

            clean_file_name_new = clean_file_name.split('.')[0]
            noisy_file_dir = os.path.join(DB_dir, train_mode, noise_list[random_noise], keyword, clean_file_name)
            file_dir = line.strip()
            f_wav_scp.writelines(wav_id + ' ' + noisy_file_dir + '\n')
            f_text.writelines(wav_id + ' ' + clean_file_dir + '\n')
    f_wav_scp.close()
    f_text.close()
