# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import torch
import torchvision
import os
import sys
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath
from video_backbone.TSP.common import utils
from video_backbone.TSP.common import transforms as T
from video_backbone.TSP.common import utils as tsp_utils
from video_backbone.TSP.models.model import Model as TSPModel
from video_backbone.TSP.train.untrimmed_video_dataset import UntrimmedVideoDataset
from video_backbone.TSP.common.scheduler import WarmupMultiStepLR
from torchvision.datasets.samplers import DistributedSampler
from itertools import chain
# from video_backbone.TSP.common.scheduler import WarmupMultiStepLR
# from TSPmodel import Model
# from model_TA import NewModel
# from NewDataset import NewDataset
from data.video_dataset import PropSeqDataset, collate_fn



from NewEval_utils import evaluate
# import new_opts as opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from NewDataset import collate_fn
from collections import OrderedDict

from ete import opts
# from video_backbone.TSP.extract_features import opts
# from ete.train import train, setup_pdvc_env, setup_tsp_env, saving_checkpoint, load_saved_info
from ete.eval import extract_features_ete

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))

MODEL_URLS = {
    # main TSP models
    'r2plus1d_34-tsp_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth',
    'r2plus1d_34-tsp_on_thumos14': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_thumos14-max_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_4-e6a30b2f.pth',

    # main TAC baseline models
    'r2plus1d_34-tac_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.002-epoch_5-98ccac94.pth',
    'r2plus1d_34-tac_on_thumos14': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_thumos14-backbone_lr_0.00001-fc_lr_0.002-epoch_3-54b5c8aa.pth',
    'r2plus1d_34-tac_on_kinetics': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_kinetics-0547130e.pth',

    # other models from the GVF and backbone architecture ablation studies
    'r2plus1d_34-tsp_on_activitynet-avg_gvf': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-avg_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-8b74eaa2.pth',
    'r2plus1d_34-tsp_on_activitynet-no_gvf': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-no_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-fb38fdd2.pth',

    'r2plus1d_18-tsp_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-22835b73.pth',
    'r2plus1d_18-tac_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.004-epoch_5-9f56941a.pth',
    'r2plus1d_18-tac_on_kinetics': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_kinetics-76ce975c.pth',

    'r3d_18-tsp_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-85584422.pth',
    'r3d_18-tac_on_activitynet': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_activitynet-backbone_lr_0.001-fc_lr_0.01-epoch_5-31fd6e95.pth',
    'r3d_18-tac_on_kinetics': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_kinetics-dcd952c6.pth',
}


def load_saved_info(opt, logger, save_folder):
    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # continue training
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain', 'metadata_csv_extract']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                   vars(opt).get(opt_name)))

    return saved_info


def setup_tsp_env(args, save_folder):
    print('############## Setting up TSP environment...')

    tsp_utils.init_distributed_mode(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.backbone_output_dir, exist_ok=True)

    train_dir = os.path.join(args.root_dir, args.train_subdir)
    valid_dir = os.path.join(args.root_dir, args.valid_subdir)

    print('LOADING DATA')
    label_mappings = []
    for label_mapping_json in args.label_mapping_jsons:
        with open(label_mapping_json) as fobj:
            label_mapping = json.load(fobj)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))

    # train_augmentations = trans.create_video_transform(
    #     'train', aug_type='augmix', crop_size=112, num_samples=None,
    #     video_mean= (0.43216, 0.394666, 0.37645),
    #     video_std= (0.22803, 0.22145, 0.216989),
    #     min_size= 128,
    #     max_size= 128
    # )

    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])

    transform_train = None
    transform_valid = None
    if args.backbone_tsp != 'mvit_v2_s':

        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            T.RandomHorizontalFlip(),
            normalize,
            T.RandomCrop((112, 112))
        ])

        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            normalize,
            T.CenterCrop((112, 112))
        ])

    else:
        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            normalize,
            T.RandomCrop((224, 224))
        ])

        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            normalize,
            T.CenterCrop((224, 224))
        ])

    dataset_train = UntrimmedVideoDataset(
        csv_filename=args.train_csv_filename,
        root_dir=train_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        clips_per_segment=args.clips_per_segment,
        temporal_jittering=True,
        transforms=transform_train,
        label_columns=args.label_columns,
        label_mappings=label_mappings,
        global_video_features=args.global_video_features,
        debug=args.debug)

    # dataset_valid = UntrimmedVideoDataset(
    #     csv_filename=args.valid_csv_filename,
    #     root_dir=valid_dir,
    #     clip_length=args.clip_len,
    #     frame_rate=args.frame_rate,
    #     clips_per_segment=args.clips_per_segment,
    #     temporal_jittering=False,
    #     transforms=transform_valid,
    #     label_columns=args.label_columns,
    #     label_mappings=label_mappings,
    #     global_video_features=args.global_video_features,
    #     debug=args.debug)

    # print('CREATING DATA LOADERS')
    sampler_train = DistributedSampler(dataset_train, shuffle=True) if args.distributed else None
    # sampler_valid = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(sampler_train is None), sampler=sampler_train,
        num_workers=args.clip_workers, pin_memory=True)

    # data_loader_valid = torch.utils.data.DataLoader(
    #     dataset_valid, batch_size=args.batch_size, shuffle=False, sampler=sampler_valid,
    #     num_workers=args.workers, pin_memory=True)

    print('CREATING VIDEO BACKBONE MODEL')

    model = TSPModel(backbone=args.backbone_tsp, num_classes=[len(l) for l in label_mappings],
                  num_heads=len(args.label_columns), concat_gvf=args.global_video_features is not None)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # targets with -1 indicate missing label

    backbone_params = None
    params = None
    fc_params = model.fc.parameters() if len(args.label_columns) == 1 \
        else chain(model.fc1.parameters(), model.fc2.parameters())

    if model.backbone != 'mvit_v2_s':
        backbone_params = chain(model.features.layer1.parameters(),
                                model.features.layer2.parameters(),
                                model.features.layer3.parameters(),
                                model.features.layer4.parameters())

        params = [
            {'params': model.features.stem.parameters(), 'lr': 0, 'name': 'stem'},
            {'params': backbone_params, 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
            {'params': fc_params, 'lr': args.fc_lr * args.world_size, 'name': 'fc'}
        ]

    else:
        backbone_params = chain(
            model.features.conv_proj.parameters(),
            model.features.pos_encoding.parameters(),
            model.features.blocks.parameters(),
            model.features.norm.parameters(),
            model.features.head.parameters()
        )
        params = [
            {'params': model.features.parameters(), 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
            {'params': fc_params, 'lr': args.fc_lr * args.world_size, 'name': 'fc'}
        ]

    optimizer = torch.optim.SGD(
        params, momentum=args.momentum, weight_decay=args.backbone_weight_decay
    )

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    if args.pretrained_tsp_path and (not args.start_from):
        print(f"Create a model from TSP's pretrained weights")
        pretrained_state_dict = torch.load(args.pretrained_tsp_path, map_location='cpu')['model']
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' not in k}
        state_dict = model.state_dict()
        pretrained_state_dict['fc1.weight'] = state_dict['fc1.weight']
        pretrained_state_dict['fc1.bias'] = state_dict['fc1.bias']
        pretrained_state_dict['fc2.weight'] = state_dict['fc2.weight']
        pretrained_state_dict['fc2.bias'] = state_dict['fc2.bias']
        model.load_state_dict(pretrained_state_dict)

        fc_params = model.fc.parameters() if len(args.label_columns) == 1 \
            else chain(model.fc1.parameters(), model.fc2.parameters())

        if model.backbone != 'mvit_v2_s':
            backbone_params = chain(model.features.layer1.parameters(),
                                    model.features.layer2.parameters(),
                                    model.features.layer3.parameters(),
                                    model.features.layer4.parameters())

            params = [
                {'params': model.features.stem.parameters(), 'lr': 0, 'name': 'stem'},
                {'params': backbone_params, 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
                {'params': fc_params, 'lr': args.fc_lr * args.world_size, 'name': 'fc'}
            ]

        else:
            params = [
                {'params': model.features.parameters(), 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
                {'params': fc_params, 'lr': args.fc_lr * args.world_size, 'name': 'fc'}
            ]

        optimizer = torch.optim.SGD(
            params, momentum=args.momentum, weight_decay=args.backbone_weight_decay
        )
        args.start_epoch = 0
        warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
        lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
        lr_scheduler = WarmupMultiStepLR(
            optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
            warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.start_from:
        print(f'Resuming TSP from checkpoint {args.start_from}')
        if args.start_from_mode == 'best':
            checkpoint = torch.load(os.path.join(save_folder, 'model-best.pth'))["video_backbone"]
        elif args.start_from_mode == 'last':
            checkpoint = torch.load(os.path.join(save_folder, 'model-last.pth'))["video_backbone"]
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    return (device, train_dir, valid_dir, transform_train, transform_valid, label_mappings,
            model, criterion, optimizer, lr_scheduler, model_without_ddp)


def extract_train_features(opt):
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, 'train.log')

    saved_info = load_saved_info(opt, logger, save_folder)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)

    tsp_env = setup_tsp_env(opt, save_folder)
    (device, train_dir, valid_dir, transform_train, transform_valid, label_mappings,
     vb_model, vb_criterion, vb_optimizer, vb_lr_scheduler, model_without_ddp) = tsp_env

    extract_features_ete(opt, save_folder, iteration, opt.metadata_csv_extract, vb_model, transform_train, subdir=opt.train_subdir)

    return saved_info


if __name__ == '__main__':
    # opt for PDVC
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # to avoid OMP problem on macos
    extract_train_features(opt)
    # train(opt)

