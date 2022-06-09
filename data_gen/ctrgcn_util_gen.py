import numpy as np
import argparse
from tqdm import tqdm


def main(args):

    with open(args.skes_name_file, 'r') as f:
        skes_names = f.read().splitlines()

    actions = np.loadtxt(args.actions_list, dtype=np.int)

    setup = []
    camera = []
    label = []
    replication = []
    performer = []
    file_name = []

    for name in tqdm(skes_names):
        if int(name[17:20]) in actions:
            file_name.append(name)
            setup.append(int(name[1:4]))
            camera.append(int(name[5:8]))
            performer.append(int(name[9:12]))
            replication.append(int(name[13:16]))
            label.append(np.where(actions == int(name[17:20]))[0][0] + 1)

    np.savetxt(args.out_path + "camera.txt", camera, fmt='%d')
    np.savetxt(args.out_path + "setup.txt", setup, fmt='%d')
    np.savetxt(args.out_path + "label.txt", label, fmt='%d')
    np.savetxt(args.out_path + "performer.txt", performer, fmt='%d')
    np.savetxt(args.out_path + "replication.txt", replication, fmt='%d')
    np.savetxt(args.out_path + "skes_available_name2.txt", file_name, fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CTR-GCN Utility Files Generator')

    parser.add_argument('--skes_name_file', default='../CTR-GCN/data/ntu/statistics/skes_available_name.txt')
    parser.add_argument('--actions_list', default='./utils/actions.txt')
    parser.add_argument('--out_path', default='../CTR-GCN/data/ntu/statistics/')

    args = parser.parse_args()

    main(args)

