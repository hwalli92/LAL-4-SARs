import sys
import os
import torch
import csv
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from FACIL.networks.network import LLL_Net
from FACIL.datasets.ntu_dataset import NTUDataset


def main(argv=None):
    # Arguments
    parser = argparse.ArgumentParser(description="Incremental Learning Eval")

    parser.add_argument("--gpu", type=int, default=1, help="GPU (default=%(default)s)")
    parser.add_argument("--config", default="./config/IL_nt_test_config.yaml", type=str, help="Network architecture used (default=%(default)s)")
    parser.add_argument("--network", default="ctrgcn", type=str, help="Network architecture used (default=%(default)s)", metavar="NETWORK")

    args, extra_args = parser.parse_known_args(argv)
    with open(args.config, "r") as f:
        config_args = yaml.load(f)

    default_arg = config_args["task_args"]
    default_arg.update(config_args[args.network])

    parser.set_defaults(**default_arg)
    args, extra_args = parser.parse_known_args(argv)
    args.model_path = os.path.expanduser(args.model_path)

    print('=' * 108)
    print('Arguments: ')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Init Device
    device = init_device(args)

    # Load Data
    test_loader = load_data(args)

    # Load Model
    model, bias_layers = load_model(args, device)

    # Eval Model
    with torch.no_grad():
        total_hits, total_num = 0, 0
        label_list = []
        pred_list = []
        model.eval()
        process = tqdm(test_loader, dynamic_ncols=True)
        for idx, (skeletons, targets) in enumerate(process):
            label_list.append(targets)
            # Forward model
            outputs = model(skeletons.to(device))
            # Using BiC IL approach
            if bias_layers is not None:
                outputs = bic_forward(device, bias_layers, outputs)

            pred = torch.zeros_like(targets.to(device))
            pred = torch.cat(outputs, dim=1).argmax(1)
            pred_list.append(pred.data.cpu().numpy())

            hits = (pred == targets).float()

            # Log
            total_hits += hits.sum().item()
            total_num += len(targets)

    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    confusion = confusion_matrix(label_list, pred_list)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = list_diag / list_raw_sum
    with open('{}/acc_each_class.csv'.format(args.model_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(each_acc)
        writer.writerows(confusion)

    print('Total Accuracy = {:.2f}%'.format((total_hits /total_num) * 100))

def init_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = "cuda"
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = "cpu"

def load_data(args):
    npz_data = np.load(args.data_path)
    x = npz_data['x_test']
    y = np.where(npz_data['y_test'] > 0)[1]
    N, T, J = x.shape
    J = int(J/6)
    x = x.reshape((N, T, 2, J, 3)).transpose(0, 4, 1, 3, 2)

    data = {'x': [], 'y': []}

    for skeleton, action in zip(x, y):
        for j in args.joints_to_mask:
            skeleton[:,:,j,:] = 0
        action = args.action_list.index(action)
        data['x'].append(skeleton)
        data['y'].append(action)

    test_data = NTUDataset(data, [], **args.test_data_args)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=False)

    return test_loader

def load_model(args, device):
    if args.network == "ctrgcn":
        sys.path.append("./CTR-GCN")
        mod_str, _sep, class_str = args.model.rpartition(".")
        __import__(mod_str)
        net = getattr(sys.modules[mod_str], class_str)
        init_model = net(**args.model_args)
        init_model.head_var = "fc"

    model = LLL_Net(init_model, remove_existing_head=True)

    pretrained = torch.load(args.model_path + "models/task9.ckpt")

    #Create Model Heads
    for i in range(args.num_tasks):
        model.add_head(args.cpertask)

    model.set_state_dict(pretrained["model"])
    model.to(device)

    if pretrained["bias_layers"] is not None:
        from FACIL.approach.bic import Appr, BiasLayer
        bias_layers = []
        for layer in pretrained["bias_layers"]:
            bias_layer = BiasLayer(device)
            bias_layer.load_state_dict(layer)
            bias_layers.append(bias_layer.to(device))
    else:
        bias_layers = None

    return model, bias_layers

def bic_forward(device, bias_layers, outputs):
    """BIC Forward Utility function --- inspired by https://github.com/sairin1202/BIC"""
    bic_outputs = []
    for m in range(len(outputs)):
        bic_outputs.append(bias_layers[m](outputs[m]))
    return bic_outputs

if __name__ == "__main__":
    main()
