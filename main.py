import os.path as osp
import os
import argparse
import subprocess

def main(args):

    if args.model == "ctrgcn":

        if args.benchmark == 'xsub':
            work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 CTR-GCN/main.py --config config/ctrgcn/nturgbd-cross-subject/train_{}.yaml --work-dir {} --device {}'.format(args.modality, work_dir, args.device), shell=True)

        elif args.benchmark == 'xview':
            work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 CTR-GCN/main.py --config config/ctrgcn/nturgbd-cross-view/train_{}.yaml --work-dir {} --device {}'.format(args.modality, work_dir, args.device), shell=True)

        else:
            print("Invalid Benchmark")

    elif args.model == "msg3d":

        if args.benchmark == 'xsub':
            work_dir = args.work_dir + "ntu/{}/msg3d_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 MS-G3D/main.py --config config/msg3d/nturgbd-cross-subject/train_{}.yaml --work-dir {} --device {}'.format(args.modality, work_dir, args.device), shell=True)

        elif args.benchmark == 'xview':
            work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
            subprocess.call('python3 MS-G3D/main.py --config config/msg3d/nturgbd-cross-view/train_{}.yaml --work-dir {} --device {}'.format(args.modality, work_dir, args.device), shell=True)

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

    else:
        print("Invalid Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Action Recognition Interface')

    parser.add_argument('--model', required=True)
    parser.add_argument('--work-dir', default='./work_dir/')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--phase', default='train')
    parser.add_argument('--benchmark', default='xsub')
    parser.add_argument('--modality', default='joint')
#    parser.add_argument()
#    parser.add_argument()

    args = parser.parse_args()

    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)

    main(args)
