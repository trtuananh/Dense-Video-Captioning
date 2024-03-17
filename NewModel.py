import torch
from torch import nn
from pdvc.pdvc import build
from TSPmodel import Model
from torchvision.io import read_video

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


    def forward(self, x, alphas=None, eval_mode=False):

        del x['video_gvf']
        
        dt = x
        
        x = dt['video_segment']     # [(start, end), ...]
        video_feature = []
        T = len(x)
        filename = dt['video_filename']
        los = 0
        
        while(len(x) > 0):
            clips = self.get_clips(x[:self.args.in_batch_size], filename, eval_mode).to(self.device)
            logits, clip_features = self.tspModel.forward(clips, gvf=None, return_features=True)     # (in_batch_size, 768)
            
            video_feature.append(clip_features.detach())
            x = x[self.args.in_batch_size:]
            
            if not eval_mode:
                middle_target = dt['video_action-label'][:self.args.in_batch_size].view(1).to(self.device)
                head_loss = self.tspCriterion(logits[0], middle_target)
                los += alphas[0] * head_loss

                # dt['video_action-label'].pop(0)
                dt['video_action-label'] = dt['video_action-label'][self.args.in_batch_size:]

        
        
        # dt['video_tensor'] = torch.vstack(video_feature).view(1, T, 768) # (1, T, 768)
        dt['video_tensor'] = torch.vstack(video_feature).unsqueeze(0)
        
        if not eval_mode:
            for param in self.tspModel.parameters():
                param.grad = None
                

        del dt['video_action-label']
        del dt['video_segment']
        
        output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)
        
        return output, loss, los
        

    def get_clips(self, segments, filename, eval_mode):
        lst = []

        for clip_t_start, clip_t_end in segments:
            # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
            vframes = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            
            if eval_mode:
                vframes = self.transforms_valid(vframes)
            
            else:
                vframes = self.transforms_train(vframes)

            lst.append(vframes)

        return torch.stack(lst)         # (in_batch_size, C, clip_length, H, W)
