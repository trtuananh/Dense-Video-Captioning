import torch
import numpy as np
import os
from torch import nn
from pdvc.pdvc import build
from TSPmodel import Model
from torchvision.io import read_video
from video_backbone.untrimmed_video_dataset_2 import _resample_video_idx

class NewModel(nn.Module):

    def __init__(self, backbone, num_classes, num_heads, args, concat_gvf, device, transforms_train=None, transforms_valid=None):

        super(NewModel, self).__init__()
        self.tspModel = Model(backbone=backbone, num_classes=num_classes, num_heads=num_heads, concat_gvf=concat_gvf)
        self.pdvcModel, self.pdvcCriterion, self.pdvcPostprocessor = build(args)
        self.feature_size = self.tspModel.feature_size
        self.args = args
        self.tspCriterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = device
        self.transforms_train = transforms_train
        self.transforms_valid = transforms_valid


    def forward(self, x, alphas=None, eval_mode=False, weight_dict=None):

        del x['video_gvf']
        del x['video_action-label']
        del x['video_temporal-region-label']
        
        dt = x

        in_batch_size = self.args.in_batch_size
        if eval_mode:
            in_batch_size = self.args.in_batch_size_valid

        
        x = dt['video_segment']     # [(start, end), ...]
        T = len(x)
        filename = dt['video_filename']
        los = 0
        ptr = 0
        vid_feature = self.get_vid_features(filename).to(self.device)
        
        for i in range(0, vid_feature.shape[0], in_batch_size):
            clips = self.get_clips(x[i : i + in_batch_size], filename, dt['video_fps'], eval_mode).to(self.device)
            _, clip_features = self.tspModel.forward(clips, gvf=None, return_features=True)     # (in_batch_size, 768)
            
            vid_feature[i : i + in_batch_size] = clip_features
        
            dt['video_tensor'] = vid_feature

            self.pdvcModel.eval()
            output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)

            if not eval_mode:
                for param in self.tspModel.parameters():
                    param.grad = None

                final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
                final_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip)


        del dt['video_segment']
        
        return output, loss, los
        

    def get_clips(self, segments, filename, fps, eval_mode):
        lst = []

        for clip_t_start, clip_t_end in segments:
            # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
            vframes, _, _ = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            idxs = _resample_video_idx(self.args.clip_len, fps, self.args.frame_rate)
            vframes = vframes[idxs][:self.args.clip_len]
            
            if eval_mode:
                vframes = self.transforms_valid(vframes)
            
            else:
                vframes = self.transforms_train(vframes)

            lst.append(vframes)

        return torch.stack(lst)         # (in_batch_size, C, clip_length, H, W)


    def get_vid_features(self, filename):
            filename = os.path.join('data/yc2/features/tsp_mvitv2', filename[-17:-4] + '.npy')
            vid_features = np.load(filename)
            vid_features = torch.from_numpy(vid_features)     # (T, 768)

            return vid_features     # (T, 768)