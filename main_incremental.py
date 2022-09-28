import sys
import os
import time
import torch
import argparse
import importlib
import yaml
import numpy as np
from functools import reduce

#sys.path.append('./FACIL/src')
import FACIL.utils as utils
import FACIL.approach
from FACIL.loggers.exp_logger import MultiLogger
from FACIL.datasets.data_loader import get_loaders
from FACIL.datasets.dataset_config import dataset_config
from FACIL.last_layer_analysis import last_layer_analysis
from FACIL.networks import tvmodels, allmodels, set_tvmodel_head_var


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--config', default='./config/IL_config.yaml', type=str,
                        help='Network architecture used (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['ntu'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=5, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='ctrgcn', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=FACIL.approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    with open(args.config, 'r') as f:
        config_args = yaml.load(f)

    default_arg = config_args['task_args']
    default_arg.update(config_args[args.network])
    key = vars(args).keys()
#    for k in default_arg.keys():
#        if k not in key:
#            print('WRONG ARG: {}'.format(k))
#            assert (k in key)

    parser.set_defaults(**default_arg)
    args, extra_args = parser.parse_known_args(argv)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    from FACIL.networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    elif args.network == "ctrgcn":
        sys.path.append('./CTR-GCN')
        mod_str, _sep, class_str = args.model.rpartition('.')
        __import__(mod_str)
        net = getattr(sys.modules[mod_str], class_str)
        init_model = net(**args.model_args)
        init_model.head_var = 'fc'
    elif args.network == "msg3d":
        sys.path.append('./MS-G3D')
        mod_str, _sep, class_str = args.model.rpartition('.')
        __import__(mod_str)
        net = getattr(sys.modules[mod_str], class_str)
        init_model = net(**args.model_args)
        init_model.head_var = 'fc'
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from FACIL.approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='FACIL.approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    if 'ntu' in args.datasets:
        from FACIL.datasets.exemplars_dataset_ntu import ExemplarsDataset
        Appr_ExemplarsDataset = ExemplarsDataset
    else:
        from FACIL.datasets.exemplars_dataset import ExemplarsDataset
        Appr_ExemplarsDataset = Appr.exemplars_dataset_class()

    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    if 'ntu' in args.datasets and args.cpertask == 'None':
        trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory, cpertask=None, skip_tasks=args.skip_tasks,
                                                              train_args=args.train_data_args, test_args=args.test_data_args)
    elif 'ntu' in args.datasets:
        trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory, cpertask=args.cpertask, skip_tasks=args.skip_tasks,
                                                              train_args=args.train_data_args, test_args=args.test_data_args)
    else:
        trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset and 'ntu' in args.datasets:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices, args.train_data_args,
                                                                 **appr_exemplars_dataset_args.__dict__)
    elif Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print(taskcla)
    trn_tasks = max_task - len(args.skip_tasks)
    acc_taw = np.zeros((trn_tasks, trn_tasks))
    acc_tag = np.zeros((trn_tasks, trn_tasks))
    forg_taw = np.zeros((trn_tasks, trn_tasks))
    forg_tag = np.zeros((trn_tasks, trn_tasks))
    trn_counter = 0
    for t, (_, ncla, trn_tsk) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add/Modify head for current task
        if t in args.skip_tasks:
            dest_task = args.dest_tasks[args.moved_tasks.index(t)]
            print("Adding Actions of Task {} to Task {}".format(t, dest_task))
            net.modify_head(dest_task, taskcla[dest_task][1]+1)
        else:
            net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        appr.train(trn_counter, trn_loader[t], val_loader[t], trn_tsk)
        print('-' * 108)

        # Test
        if trn_tsk:
            for u in range(trn_counter + 1):
                test_loss, acc_taw[trn_counter, u], acc_tag[trn_counter, u] = appr.eval(u, tst_loader[u])
                if u < trn_counter:
                    forg_taw[trn_counter, u] = acc_taw[:trn_counter, u].max(0) - acc_taw[trn_counter, u]
                    forg_tag[trn_counter, u] = acc_tag[:trn_counter, u].max(0) - acc_tag[trn_counter, u]
                print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                      '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                     100 * acc_taw[trn_counter, u], 100 * forg_taw[trn_counter, u],
                                                                     100 * acc_tag[trn_counter, u], 100 * forg_tag[trn_counter, u]))
                logger.log_scalar(task=trn_counter, iter=u, name='loss', group='test', value=test_loss)
                logger.log_scalar(task=trn_counter, iter=u, name='acc_taw', group='test', value=100 * acc_taw[trn_counter, u])
                logger.log_scalar(task=trn_counter, iter=u, name='acc_tag', group='test', value=100 * acc_tag[trn_counter, u])
                logger.log_scalar(task=trn_counter, iter=u, name='forg_taw', group='test', value=100 * forg_taw[trn_counter, u])
                logger.log_scalar(task=trn_counter, iter=u, name='forg_tag', group='test', value=100 * forg_tag[trn_counter, u])
            trn_counter+=1
        else:
            tst_loader[dest_task] = torch.utils.data.DataLoader(tst_loader[dest_task].dataset + tst_loader[trn_counter].dataset,
                                                                batch_size=tst_loader[dest_task].batch_size,
                                                                shuffle=True,
                                                                num_workers=tst_loader[dest_task].num_workers,
                                                                pin_memory=tst_loader[dest_task].pin_memory)
            tst_loader.pop(trn_counter)

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(acc_tag, name="acc_tag", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.log_result(forg_tag, name="forg_tag", step=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:trn_tasks]]], trn_tasks, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

        if args.approach == "bic":
            bias_dict = []
            for layer in appr.bias_layers:
                bias_dict.append(layer.state_dict())
            exemplars = {'exm_ds': appr.exemplars_dataset, 'val_exm_x': appr.x_valid_exemplars, 'val_exm_y': appr.y_valid_exemplars}
            logger.save_model(net.state_dict(), task=t, exemplars=exemplars, bias_layers=bias_dict)
        else:
            exemplars = {'exm_ds': appr.exemplars_dataset, 'val_exm_x': None, 'val_exm_y': None}
            logger.save_model(net.state_dict(), task=t, exemplars=exemplars)


        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
