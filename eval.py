from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import torch
import numpy as np
import time
from os.path import dirname, abspath

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from NewEval_utils import evaluate
from NewDataset import NewDataset 
from misc.utils import create_logger
from NewDataset import collate_fn
from torch.utils.data import DataLoader
from os.path import basename
from NewModel import NewModel
import pandas as pd

def create_fake_test_caption_file(metadata_csv_path):
    out = {}
    df = pd.read_csv(metadata_csv_path)
    for i, row in df.iterrows():
        out[basename(row['filename']).split('.')[0]] = {'duration': row['video-duration'], "timestamps": [[0, 0.5]], "sentences":["None"]}
    fake_test_json = '.fake_test_json.tmp'
    json.dump(out, open(fake_test_json, 'w'))
    return fake_test_json

def main(opt):
    folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    if opt.eval_mode == 'test':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    logger = create_logger(folder_path, 'val.log')
    if opt.eval_model_path:
        model_path = opt.eval_model_path
        infos_path = os.path.join('/'.join(opt.eval_model_path.split('/')[:-1]), 'info.json')
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')
        infos_path = os.path.join(folder_path, 'info.json')

    logger.info(vars(opt))

    with open(infos_path, 'rb') as f:
        logger.info('load info from {}'.format(infos_path))
        old_opt = json.load(f)['best']['opt']

    for k, v in old_opt.items():
        if k[:4] != 'eval':
            vars(opt).update({k: v})

    opt.transformer_input_type = opt.eval_transformer_input_type

    if not torch.cuda.is_available():
        opt.nthreads = 0
    # Create the Data Loader instance

    metadata_df = pd.read_csv(opt.metadata_csv_valid)
    valid_dir = os.path.join(opt.root_dir, opt.valid_subdir)
    label_mappings = []
    for label_mapping_json in opt.label_mapping_jsons:
        with open(label_mapping_json) as fobj:
            label_mapping = json.load(fobj)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))

    if opt.eval_mode == 'test':
        opt.eval_caption_file = create_fake_test_caption_file(opt.test_video_meta_data_csv_path)
        opt.visual_feature_folder = opt.test_video_feature_folder
        metadata_df = pd.read_csv('visualization/videos/metadata.csv')
        valid_dir = 'visualization/videos'
        #if opt.visual_feature_type == ['tsp'] and opt.test_video_feature_folder == ['visualization/output/video_backbone/TSP/checkpoints/mvit_tsp.pth_stride_16/']:
    	    #opt.visual_feature_type = ['tsp_mvit']
    	    #opt.feature_dim=768
    	    #print(f'Hello from the other side')
    	        


    val_dataset = NewDataset(
        csv_filename=metadata_df,
        root_dir=valid_dir,
        clip_length=opt.clip_len,
        frame_rate=opt.frame_rate,
        stride=opt.stride,
        anno_file=opt.val_caption_file,
        dict_file=opt.dict_file,
        is_training=False,
        proposal_type='gt',
        opt=opt,
        transforms=None,
        global_video_feature=None,
        label_columns=opt.label_columns,
        label_mappings=label_mappings,
        dataset_type='valid'
    )
    
    loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True, collate_fn=collate_fn)


    model = NewModel(backbone=opt.backbone_tsp, num_classes=[len(l) for l in label_mappings], num_heads=len(opt.label_columns), concat_gvf=opt.global_video_features is not None, 
                     device=torch.device(opt.device), args=opt, transforms_valid=None, transforms_train=None)
    model.pdvcModel.translator = val_dataset.translator



    while not os.path.exists(model_path):
        raise AssertionError('File {} does not exist'.format(model_path))

    logger.debug('Loading model from {}'.format(model_path))
    loaded_pth = torch.load(model_path, map_location=opt.eval_device)
    epoch = loaded_pth['epoch']

    # loaded_pth = transfer(model, loaded_pth, model_path+'.transfer.pth')
    model.load_state_dict(loaded_pth['model'], strict=True)
    model.eval()

    model.to(opt.eval_device)

    if opt.eval_mode == 'test':
        out_json_path = os.path.join(folder_path, 'dvc_results.json')
        evaluate(model, model.pdvcCriterion,  model.pdvcPostprocessor, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=False, skip_lang_eval=True, visualization=opt.visualization)


    else:
        out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_alpha{}.json'.format(
            time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime()) + str(opt.id), epoch, len(loader.dataset),
            opt.ec_alpha))
        caption_scores, eval_loss = evaluate(model, model.pdvcCriterion, model.pdvcPostprocessor, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=False, skip_lang_eval=False)
        avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items() if key !='tiou'}
        avg_eval_score2 = {key: np.array(value).mean() * 4917 / len(loader.dataset) for key, value in caption_scores.items() if key != 'tiou'}

        logger.info(
            '\nValidation result based on all 4917 val videos:\n {}\n avg_score:\n{}'.format(
                                                                                       caption_scores.items(),
                                                                                       avg_eval_score))

        logger.info(
                '\nValidation result based on {} available val videos:\n avg_score:\n{}'.format(len(loader.dataset),
                                                                                           avg_eval_score2))

    logger.info('saving reults json to {}'.format(out_json_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_mode', type=str, default='eval', choices=['eval', 'test'])
    parser.add_argument('--test_video_feature_folder', type=str, nargs='+', default=None)
    parser.add_argument('--test_video_meta_data_csv_path', type=str, default=None)
    parser.add_argument('--eval_folder', type=str, required=True)
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021'])
    parser.add_argument('--eval_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--eval_proposal_type', type=str, default='gt')
    parser.add_argument('--eval_transformer_input_type', type=str, default='queries', choices=['gt_proposals', 'queries'])
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--visualization', type=str, default='no')
    opt = parser.parse_args()
	
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
