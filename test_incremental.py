import sys
import os
import time
import torch
import argparse
import importlib
import yaml
import numpy as np

sys.path.append("./FACIL/src")
from approach.bic import Appr, BiasLayer
from networks.network import LLL_Net
from datasets.ntu_dataset import NTUDataset


def main(argv=None):
    tstart = time.time()

    # Arguments
    parser = argparse.ArgumentParser(description="Incremental Learning Eval")

    parser.add_argument("--gpu", type=int, default=1, help="GPU (default=%(default)s)")
    parser.add_argument(
        "--config",
        default="./config/IL_config.yaml",
        type=str,
        help="Network architecture used (default=%(default)s)",
    )
    parser.add_argument(
        "--network",
        default="ctrgcn",
        type=str,
        help="Network architecture used (default=%(default)s)",
        metavar="NETWORK",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/ntu_bic_save_model/models/",
        help="Results path (default=%(default)s)",
    )
    parser.add_argument(
        "--data-path", default="", type=str, help="Test data path (default=%(default)s)"
    )
    parser.add_argument(
        "--joints-mask",
        default=[],
        type=int,
        choices=range(0, 25, 1),
        help="Joint numbers to mask (default=%(default)s)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of Tasks (default=%(default)s)",
    )

    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    with open(args.config, "r") as f:
        config_args = yaml.load(f)

    default_arg = config_args["task_args"]
    default_arg.update(config_args[args.network])

    # Init Device
    device = init_device(args)

    # Load Data
    test_loader = load_data(args)

    # Load Model
    model, bias_layers = load_model(args)
    appr = Appr(model, device)
    if bias_layers is not None:
        appr.bias_layers = bias_layers

    # Eval Model
    acc_taw = np.zeros(args.num_tasks)
    acc_tag = np.zeros(args.num_tasks)

    for t in range(args.num_tasks):
        loss, acc_taw[trn_counter, u], acc_tag[trn_counter, u] = appr.eval(
            t, test_loader[t]
        )

        print(
            "Test on Task {:2d}: TAw Acc={}% | TAg Acc={}%".format(
                t, 100 * acc_taw[t], 100 * acc_tag[t]
            )
        )

    def init_device(args):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = "cuda"
        else:
            print("WARNING: [CUDA unavailable] Using CPU instead!")
            device = "cpu"

    def load_data(args):
        pass

    def load_model(args, device):
        if args.network == "ctrgcn":
            sys.path.append("./CTR-GCN")
            mod_str, _sep, class_str = args.model.rpartition(".")
            __import__(mod_str)
            net = getattr(sys.modules[mod_str], class_str)
            init_model = net(**args.model_args)
            init_model.head_var = "fc"

        model = LLL_Net(init_model)

        pretrained = torch.load(args.model_path + "task9.chpt")
        model.load_state_dict(pretrained["model"])
        model.to(device)

        if pretrained["bias_layers"] is not None:
            bias_layers = []
            for layer in pretrained["bias_layers"]:
                bias_layer = BiasLayer()
                bias_layer.load_state_dict(layer)
                bias_layer.to(device)
                bias_layers.append(bias_layer)
        else:
            bias_layers = None

        return model, bias_layers


if __name__ == "__main__":
    main()
