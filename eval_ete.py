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
# from torchvision.datasets.samplers import DistributedSampler
from itertools import chain
# from video_backbone.TSP.common.scheduler import WarmupMultiStepLR
from TSPmodel import Model
from model_TA import NewModel
from NewDataset import NewDataset
from data.video_dataset import PropSeqDataset, collate_fn

from NewEval_utils import evaluate
# import new_opts as opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from NewDataset import collate_fn
from pdvc.pdvc import build
from collections import OrderedDict

from ete import opts
from ete.train import train, setup_pdvc_env, setup_tsp_env, saving_checkpoint

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


def main(opt):
    pdvc_env = setup_pdvc_env(opt)
    (save_folder, logger, tf_writer, saved_info,
        epoch, iteration, video_idx, best_val_score, val_result_history, loss_history, lr_history,
        model, criterion, postprocessors, optimizer, lr_scheduler) = pdvc_env

    tsp_env = setup_tsp_env(opt, save_folder)

    train_dataset = PropSeqDataset(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   opt.dict_file, True, 'gt',
                                   opt)

    model.translator = train_dataset.translator

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message('Start training !', logger)

    weight_dict = criterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    saving_checkpoint(opt, save_folder, logger, tf_writer, saved_info,
                      epoch, iteration, video_idx, best_val_score, val_result_history, loss_history, lr_history,
                      model, criterion, postprocessors, optimizer, lr_scheduler, tsp_env, eval_flag=True)

    return saved_info


if __name__ == '__main__':
    # opt for PDVC
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # to avoid OMP problem on macos
    main(opt)
    # train(opt)

