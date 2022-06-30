#!/usr/bin/env bash

echo Preparing utility files

cp utils/ntu_generate_data.py ../VA_NN/data/
cd ../VA_NN/

echo Creating xview and xsub datasets
python3 data/ntu_generate_data.py --data_path /media/ntfs-data/datasets/ntu/nturgb+d_60_skeletons/ --actions_list ../data_gen/utils/actions.txt
