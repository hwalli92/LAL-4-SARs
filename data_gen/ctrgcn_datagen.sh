#!/usr/bin/env bash

echo Preparing utility files
python3 ctrgcn_util_gen.py

cp utils/get_raw_skes_data.py ../CTR-GCN/data/ntu/
cd ../CTR-GCN/data/ntu/

echo Extracting raw skeleton files
python3 get_raw_skes_data.py

echo Denoising skeleton data
python3 get_raw_denoised_data.py

echo Creating xview and xsub datasets
python3 seq_transformation.py
