#!/bin/bash

python src/reid/train.py -c 'configs/hrnetW48.yaml' -o CHECKPOINT_PATH="" > logs/reid_hrnet_$(date +%F-%H-%M-%S).log 2>&1

python src/reid/train.py -c 'configs/transformer.yaml' -o CHECKPOINT_PATH="" > logs/reid_transformer_$(date +%F-%H-%M-%S).log 2>&1

python src/reid/train.py -c 'configs/transformer_local.yaml' -o CHECKPOINT_PATH="" > logs/reid_transformer_local_$(date +%F-%H-%M-%S).log 2>&1