import os.path as osp
import os
import argparse
import subprocess

def main(args):

    if args.model == "ctrgcn":

        if args.benchmark == 'xsub':
            if args.modality is None:
                work_dir = args.work_dir + "ntu/{}/ctrgcn_joint".format(args.benchmark)
                subprocess.call('python3 CTR-GCN/main.py --config config/nturgbd-cross-subject/default.yaml --work-dir {} --device {}'.format(work_dir, args.device), shell=True)

            else:
               d work_dir = args.work_dir + "ntu/{}/ctrgcn_{}".format(args.benchmark, args.modality)
                subprocess.call('python3 CTR-GCN/main.py --config config/nturgbd-cross-subject/default.yaml --train_feeder_args {}=True --test_feeder_args {}=True --work-dir {} --device {}'.format(
                                 args.modality, args.modality, work_dir, args.device),
                                 shell=True)

    elif args.model == "msg3d":

        work_dir = args.work_dir + "ntu/{}/msg3d_{}".format(args.benchmark, args.modality)
        print(work_dir)

    else:
        print("Invalid Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Action Recognition Interface')

    parser.add_argument('--model', required=True)
    parser.add_argument('--work-dir', default='./work_dir/')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--phase', default='train')
    parser.add_argument('--benchmark', default='xsub')
    parser.add_argument('--modality', default=None)
#    parser.add_argument()
#    parser.add_argument()

    args = parser.parse_args()

    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)

    main(args)
