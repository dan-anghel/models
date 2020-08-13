#!/bin/bash

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

current_time=$(date "+%Y%m%d%H%M%S")

#python train.py \
#  --train_file_pattern=/media/gld/tfrecord/tfrecord_clean_relabeled_split_rehashed/train-*-of-00128 \
#  --validation_file_pattern=/media/gld/tfrecord/tfrecord_clean_relabeled_split_rehashed/validation-*-of-00128 \
#  --imagenet_checkpoint=gs://dananghel-delf/imagenet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#  --dataset_version=gld_v2_clean \
#  --max_iters=50000 \
#  --use_augmentation \
#  --block3_strides \
#  --batch_size=256 \
#  --logdir=gs://dananghel-delf/training/${current_time}_gldv2_reshufled_500_valbatch_p100 1>train_log.txt 2>&1 &

python train.py \
  --train_file_pattern=/media/gld/tfrecord/tfrecord_clean_relabeled_split_rehashed/train-*-of-00128 \
  --validation_file_pattern=/media/gld/tfrecord/tfrecord_clean_relabeled_split_rehashed/validation-*-of-00128 \
  --imagenet_checkpoint=gs://dananghel-delf/imagenet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  --dataset_version=gld_v2_clean \
  --max_iters=1000 \
  --use_augmentation \
  --block3_strides \
  --batch_size=256 \
  --delg_global_features \
  --logdir=gs://dananghel-delf/training/${current_time}_gldv2_delg_reshufled_500_valbatch_p100 1>train_log.txt 2>&1 &
