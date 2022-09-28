# LAL-4-SARs

This repo is the implementation of the Master's Thesis work, Lifelong Action Learning for Socially Assistive Robots, for the Master's of Autonomous Systems degree at Hochschule Bonn-Rhein-Sieg. This repo contains the 5 submodules needed for the action recognition and incremental learning files to work. In order to clone this repo with the submodules use the below command:

`git clone --recurse-submodules https://github.com/hwalli92/LAL-4-SARs.git`

## Prerequisites

Each submodule has their own set of dependecies and prerequisites, please review the README for each submodule to install the correct requirements. 

## Data Preparation

Data preparation for each action recognition module is different and thus follow the below steps for each.

<details>
  <summary> CTR-GCN Network </summary>

1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU RGB+D 60 dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/utils/get_raw_skes_data.py#L137
3. Run the following commands to generate the data

```
cd data_gen/
./ctrgcn_datagen.sh
```
</details>

<details>
  <summary> MS-G3D Network </summary>
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU RGB+D 60 dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/msg3d_datagen.sh#L9
3. Run the following commands to generate the data

```
cd data_gen/
./msg3d_datagen.sh
```
  
</details>

<details>
  <summary> EfficientGCN Network </summary>
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Update number of actions classes here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/utils/__init__.py#L8
2. Add path to NTU RGB+D 60 dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/msg3d_datagen.sh#L9
3. Run the following commands to generate the data

```
cd data_gen/
./efficientgcn.sh
cd ../EfficientGCNv1/
python3 main.py -c <config> -gd
```
Refer to the EfficientGCN README file for the config # in the above argument.

</details>
  
<details>
  <summary> VA-NN Network </summary>
1. Update action list: [actions.txt](data_gen/utils/actions.txt)
2. Add path to NTU RGB+D 60 dataset here: https://github.com/hwalli92/LAL-4-SARs/blob/07949985e0f596714a102ea39338cd000445e0a6/data_gen/va_datagen.sh#L9
3. Run the following scripts to generate the data

```
cd data_gen/
./va_datagen.sh
```

</details>

## Action Recognition Network Training 

Run the action recognition training with:
```
python3 main_recognition.py --model <network>
```
followed by general options:

* `--work-dir`: working directory where model will be saved (default='./work_dir/')
* `--device`: index of GPU to run the experiment on (default=1)
* `--phase`: must be train or test (default='train')
* `--benchmark`: must be xsub or xview (default='xsub')
* `--modality`: must be joint or bone (default='joint')
  
## Incremental Learning Training

Run the incremental learning training with:
```
python3 main_incremental.py
```
followed by general options:

* `--gpu`: index of GPU to run the experiment on (default=1)
* `--results-path`: path where results are stored (default='./results')
* `--config`: config file for IL process (default='./config/IL_config.yaml')
* `--approach`: learning approach used (default='finetuning') [[more](FACIL/FACIL/approach)]
* `--datasets`: dataset or datasets used (default=['ntu'])
* `--network`: network architecture used (default='ctrgcn')
* `--num-exemplars`: Fixed memory, total number of exemplars (default=0)
* `--num-exemplars-per-class`: Growing memory, number of exemplars per class (default=0)
* `--exemplar-selection`: Exemplar selection method (default='random')
  
## Incremental Learning Testing

Run the incremental learning training with:
```
python3 test_incremental.py
```
followed by general options:
  
* `--gpu`: index of GPU to run the experiment on (default=1)
* `--config`: config file for IL process (default='./config/IL_ntu_test_config.yaml')
* `--network`: network architecture used (default='ctrgcn')

## Incremental Learning Results GUI
  
To view the incremental TAw and TAg accuracy and forget percentage plots, run the results plotter with:
```
python3 results_plotter.py
```

Ensure the name of your results folder is listed in this config file: [results_config.yaml](results/results_config.yaml)
