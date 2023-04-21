#!/bin/bash

python src/reid/inference.py -c 'configs/hrnetW48.yaml' -o LABEL_FOLDER='outputs/tracking_results'
python src/reid/inference.py -c 'configs/transformer.yaml' -o LABEL_FOLDER='outputs/tracking_results'
python src/reid/inference.py -c 'configs/transformer_local.yaml' -o LABEL_FOLDER='outputs/tracking_results'