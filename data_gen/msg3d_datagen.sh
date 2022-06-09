#!/usr/bin/env bash

echo Preparing utility files

cp utils/ntu_gendata.py ../MS-G3D/data_gen
cd ../MS-G3D/data_gen

echo Creating xview and xsub datasets
python3 ntu_gendata.py --data_path /media/ntfs-data/datasets/ntu/nturgb+d_60_skeletons/ --actions_list ../../data_gen/utils/actions.txt

echo Generating bone data
python3 gen_bone_data.py --dataset ntu
