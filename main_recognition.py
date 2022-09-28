import os.path as osp
import os
import argparse
import subprocess

def main(args):

    if args.model == "ctrgcn":

        if args.benchmark == 'xsub':
            work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 CTR-GCN/main.py --config config/ctrgcn/{}_{}.yaml --work-dir {} --device {}'.format(args.benchmark, args.modality, work_dir, args.device), shell=True)

        elif args.benchmark == 'xview':
            work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 CTR-GCN/main.py --config config/ctrgcn/{}_{}.yaml --work-dir {} --device {}'.format(args.benchmark, args.modality, work_dir, args.device), shell=True)

        else:
            print("Invalid Benchmark")

    elif args.model == "msg3d":

        if args.benchmark == 'xsub':
            work_dir = args.work_dir + "ntu/{}/msg3d_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 MS-G3D/main.py --config config/msg3d/{}_{}_{}.yaml --work-dir {} --device {}'.format(args.benchmark, args.phase, args.modality, work_dir, args.device), shell=True)

        elif args.benchmark == 'xview':
            work_dir = args.work_dir + "ntu/{}/msg3d_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 MS-G3D/main.py --config config/msg3d/{}_{}_{}.yaml --work-dir {} --device {}'.format(args.benchmark, args.phase, args.modality, work_dir, args.device), shell=True)

        else:
            print("Invalid Benchmark")

    elif args.model == "efficientgcn":

        if args.benchmark == '2009' or args.benchmark == '2021':
            work_dir = '../' + args.work_dir + "efficientgcn/{}".format(args.benchmark)
            subprocess.call('python3 EfficientGCNv1/main.py -c {} -gd'.format(args.benchmark), shell=True)
            subprocess.call('python3 EfficientGCNv1/main.py -c {} -g {}'.format(args.benchmark, args.device), shell=True)

        elif args.benchmark == '2010' or args.benchmark == '2022':
            work_dir = args.work_dir + "ntu/xview/efficientgcn_{}".format(args.benchmark)
            subprocess.call('python3 EfficientGCNv1/main.py --c ../config/efficientgcn/{}.yaml -gd'.format(args.benchmark), shell=True)
            subprocess.call('python3 EfficientGCNv1/main.py --c ../config/efficientgcn/{}.yaml -w {} -g {}'.format(args.benchmark, work_dir, args.device), shell=True)

        else:
            print("Invalid Benchmark")

    elif args.model == "vacnn":

        if args.benchmark == 'xsub':
            data_dir = "VA_NN/data/"
            config_dir = "VA_NN/config/"
            weights_dir = "VA_NN/weights/"
            logs_dir = "VA_NN/logs/"
            subprocess.call('python3 VA_NN/main_cnn.py --dataset_dir {} --dataset_name cs --cfg_dir {} --save_dir {} --log_dir {}'.format(data_dir, config_dir, weights_dir, logs_dir), shell=True)

        elif args.benchmark == 'xview':
            data_dir = "VA_NN/data/"
            config_dir = "VA_NN/config/"
            weights_dir = "VA_NN/weights/"
            logs_dir = "VA_NN/logs/"
            subprocess.call('python3 VA_NN/main_cnn.py --dataset_dir {} --dataset_name cv --cfg_dir {} --save_dir {} --log_dir {}'.format(data_dir, config_dir, weights_dir, logs_dir), shell=True)

        else:
            print("Invalid Benchmark")

    elif args.model == "varnn":

        if args.benchmark == 'xsub':
            data_dir = "VA_NN/data/"
            config_dir = "VA_NN/config/"
            weights_dir = "VA_NN/weights/"
            logs_dir = "VA_NN/logs/"
            subprocess.call('python3 VA_NN/main_rnn.py --dataset_dir {} --dataset_name cs --cfg_dir {} --save_dir {} --log_dir {}'.format(data_dir, config_dir, weights_dir, logs_dir), shell=True)

        elif args.benchmark == 'xview':
            data_dir = "VA_NN/data/"
            config_dir = "VA_NN/config/"
            weights_dir = "VA_NN/weights/"
            logs_dir = "VA_NN/logs/"
            subprocess.call('python3 VA_NN/main_rnn.py --dataset_dir {} --dataset_name cv --cfg_dir {} --save_dir {} --log_dir {}'.format(data_dir, config_dir, weights_dir, logs_dir), shell=True)

        else:
            print("Invalid Benchmark")


    else:
        print("Invalid Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Action Recognition Interface')

    parser.add_argument('--model', required=True)
    parser.add_argument('--work-dir', default='./work_dir/')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--phase', default='train')
    parser.add_argument('--benchmark', default='xsub')
    parser.add_argument('--modality', default='joint')

    args = parser.parse_args()

    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)

    main(args)
