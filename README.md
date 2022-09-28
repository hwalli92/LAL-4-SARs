# LAL-4-SARs

This repo is the implementation of the Master's Thesis work, Lifelong Action Learning for Socially Assistive Robots, for the Master's of Autonomous Systems degree at Hochschule Bonn-Rhein-Sieg. This repo contains the 5 submodules needed for the action recognition and incremental learning files to work. In order to clone this repo with the submodules use the below command:

`git clone --recurse-submodules https://github.com/hwalli92/LAL-4-SARs.git`

## Prerequisites

Each submodule has their own set of dependecies and prerequisites, please review the README for each submodule to install the correct requirements. 

## Data Preparations

Data preparation for each action recognition module is different and thus follow the below steps for each.

#### CTR-GCN Network
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/utils/get_raw_skes_data.py#L137
3. Run the following commands to generate the data

```
cd data_gen/
./ctrgcn_datagen.sh
```

#### MS-G3D Network
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/msg3d_datagen.sh#L9
3. Run the following commands to generate the data

```
cd data_gen/
./msg3d_datagen.sh
```

#### EfficientGCN Network
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Update number of actions classes here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/utils/__init__.py#L8
2. Add path to NTU dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/msg3d_datagen.sh#L9
3. Run the following commands to generate the data

```
cd data_gen/
./efficientgcn.sh
cd ../EfficientGCNv1/
python3 main.py -c <config> -gd
```
Refer to the EfficientGCN README file for the config # in the above argument.

#### VA-NN Network
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/va_datagen.sh#L9
3. Run the following scripts to generate the data

```
cd data_gen/
./va_datagen.sh
```

