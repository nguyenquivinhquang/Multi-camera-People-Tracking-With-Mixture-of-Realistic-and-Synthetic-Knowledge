#!/bin/bash
DATASET_DIR="/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking"
DATASET_DIR=${DATASET_DIR}'/test'

# prepare dataset
python src/SCMT/dataspace/AICITY_test/prepapre_folder_infor.py

# Move to tracking dir
cd src/SCMT/


echo "${DATASET_DIR}"

scenes=(S003 S004 S009 S014 S021 S022 S001)

for scene in "${scenes[@]}"
do
    python run_aicity.py --scene "$scene" --feature_dir ../../output/transformer_feat --detection_dir ../../datasets/detection/Yolo
done

## Move file to tracking result
cd ../../
python src/SCMT/move2trackingres.py