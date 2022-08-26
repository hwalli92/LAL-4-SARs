import sys
import os
import time
import torch
import argparse
import importlib
import yaml
import numpy as np



def main(argv=None):
    tstart = time.time()

    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Eval')

    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--weights-path', type=str, default='results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--data-path', default='', type=str,
                        help='Test data path (default=%(default)s)')


    appr.eval(u, tst_loader[u])

if __name__ == '__main__':
    main()
