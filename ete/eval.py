from __future__ import division, print_function

import os
import torch
import torchvision
import json
import datetime
import time
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import shutil

from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from eval_utils import evaluate as pdvc_evaluate
from data.video_dataset import PropSeqDataset, collate_fn
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from data.video_dataset import PropSeqDataset, collate_fn
from collections import OrderedDict

from video_backbone.TSP.extract_features.eval_video_dataset import EvalVideoDataset

from video_backbone.TSP.common import utils
from video_backbone.TSP.common import transforms as T
from video_backbone.TSP.models.model import Model as TSPModel
from video_backbone.TSP.extract_features import extract_features

from ete.dataset import LocatedEvalDataset


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


def extract_valid_features(args, save_folder, iteration, metadata_csv_valid, model, transform_valid):
    print('START FEATURE EXTRACTION')
    # print(args)
    # print('TORCH VERSION: ', torch.__version__)
    # print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    output_dir = os.path.join(save_folder, "features", f"iter{iteration}_valid")
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('LOADING DATA')
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])

    # transform = torchvision.transforms.Compose([
    #     T.ToFloatTensorInZeroOne(),
    #     T.Resize((128, 171)),
    #     normalize,
    #     T.CenterCrop((112, 112))
    # ])

    metadata_df = pd.read_csv(metadata_csv_valid)
    shards = np.linspace(0, len(metadata_df), args.num_shards + 1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id + 1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
          f'total number of videos: {len(metadata_df)}, shard size {end_idx - start_idx} videos')

    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
                                                                     os.path.exists(os.path.join(output_dir,
                                                                                                 os.path.basename(
                                                                                                     f).split('.')[
                                                                                                     0] + '.npy')))
    metadata_df = metadata_df[metadata_df['is-computed-already'] == False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    valid_dir = os.path.join(args.root_dir, args.valid_subdir)
    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=valid_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        stride=args.stride,
        output_dir=output_dir,
        transforms=transform_valid)

    print('CREATING EXTRACT FEATURES DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.clip_batch_size_valid, shuffle=False,
        num_workers=args.clip_workers, pin_memory=True)

    extract_features.evaluate(model, data_loader, device)
    model.train()

    return output_dir


def evaluate(opt, save_folder, logger, iteration, model, criterion, postprocessors, vb_model, vb_transform):
    # Extracting Video Features
    feature_folder = extract_features_ete(opt, save_folder, iteration, opt.metadata_csv_valid, vb_model, vb_transform, subdir=opt.valid_subdir)

    # Create Valid Dataset and DataLoader
    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 [feature_folder],
                                 opt.dict_file, False, 'gt',
                                 opt)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    model.eval()
    result_json_path = os.path.join(save_folder, 'prediction',
                                    'num{}_iter{}.json'.format(
                                        len(val_dataset), iteration))
    eval_score, eval_loss = pdvc_evaluate(model, criterion, postprocessors, val_loader, result_json_path, logger=logger,
                                          alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
    model.train()

    return result_json_path, feature_folder, eval_score, eval_loss


def extract_features_ete(args, save_folder, iteration, metadata_csv, model, transform, subdir='train'):
    print('START FEATURE EXTRACTION')
    # print(args)
    # print('TORCH VERSION: ', torch.__version__)
    # print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    output_dir = os.path.join(save_folder, "features", f"iter{iteration}_{subdir}")
    os.makedirs(output_dir, exist_ok=True)

    print('LOADING DATA')

    metadata_df = pd.read_csv(metadata_csv)
    shards = np.linspace(0, len(metadata_df), args.num_shards + 1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id + 1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
          f'total number of videos: {len(metadata_df)}, shard size {end_idx - start_idx} videos')

    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
                                                                     os.path.exists(os.path.join(output_dir,
                                                                                                 os.path.basename(
                                                                                                     f).split('.')[
                                                                                                     0] + '.npy')))
    metadata_df = metadata_df[metadata_df['is-computed-already'] == False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    data_dir = os.path.join(args.root_dir, subdir)
    for filename in metadata_df['filename']:
        dataset = LocatedEvalDataset(
            video_id=filename.split('.')[0],
            vf_len=args.frame_embedding_num,
            metadata_filename=metadata_csv,
            root_dir=data_dir,
            clip_length=args.clip_len,
            frame_rate=args.frame_rate,
            stride=args.stride,
            output_dir=output_dir,
            transforms=transform)

        print('CREATING EXTRACT FEATURES DATA LOADER')
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.clip_batch_size_valid, shuffle=False,
            num_workers=args.clip_workers, pin_memory=True)

        extract_features.evaluate(model, data_loader, device)
        torch.cuda.empty_cache()
    model.train()

    return output_dir


if __name__ == '__main__':
    from opts import parse_args

    args = parse_args()
    main(args)
