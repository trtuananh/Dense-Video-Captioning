import torch
from torch import nn
import numpy as np
import os
from pdvc.pdvc import build
from TSPmodel import Model

class NewModel(nn.Module):

    def __init__(self, backbone, num_classes, num_heads, args, concat_gvf, device):

        super(NewModel, self).__init__()
        self.tspModel = Model(backbone=backbone, num_classes=num_classes, num_heads=num_heads, concat_gvf=concat_gvf)
        self.pdvcModel, self.pdvcCriterion, self.pdvcPostprocessor = build(args)
        self.feature_size = self.tspModel.feature_size
        self.args = args
        self.tspCriterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = device


    def forward(self, x, alphas=None, eval_mode=False):

        del x['video_gvf']
        
        dt = x
        
        x = dt['video_tensor'][0] 
        video_feature = None
        T = len(x)
        los = 0
        
        while(len(x) > 0):
            # clips = x[0].view(1, 3, 16, 224, 224).to(self.device)
            clips = torch.stack(x[:self.args.in_batch_size]).to(self.device)      # (in_batch_size, 3, 16, 224, 224)
            logits, clip_features = self.tspModel.forward(clips, gvf=None, return_features=True)     # (in_batch_size, 768)
            
            if video_feature is None:
                video_feature = clip_features.detach()
            else:
                video_feature = torch.cat([video_feature, clip_features.detach()], 0)

            x = x[self.args.in_batch_size:]
            
            if not eval_mode:
                middle_target = dt['video_action-label'][:self.args.in_batch_size].view(1).to(self.device)
                head_loss = self.tspCriterion(logits[0], middle_target)
                los += alphas[0] * head_loss

                # dt['video_action-label'].pop(0)
                dt['video_action-label'] = dt['video_action-label'][self.args.in_batch_size:]

        
        # if not eval_mode:
        #         for logit in video_logits:
        #             # middle_targets = [torch.tensor(dt[f'video_{col}'][0]).view(1).to(self.device) for col in self.args.label_columns]
        #             # middle_targets = [torch.tensor(dt[f'video_{col}'][:self.args.in_batch_size]).to(self.device) for col in self.args.label_columns]
        #             # for outpt, target, alpha in zip(logit, middle_targets, alphas):
        #             middle_target = torch.tensor(dt['video_action-label'][0]).view(1).to(self.device)
        #             head_loss = self.tspCriterion(logit, middle_target)
        #             los += alpha * head_loss
    
        #             # for i in range(self.args.in_batch_size):
                    
        #             dt['video_action-label'].pop(0)
        #             dt['video_temporal-region-label'].pop(0)
    
        #             print(f'Check mem 3: {torch.cuda.memory_allocated(0)}')


        
        # dt['video_tensor'] = torch.vstack(video_feature).view(1, T, 768) # (1, T, 768)
        dt['video_tensor'] = video_feature.unsqueeze(0)
        
        if not eval_mode:
            for param in self.tspModel.parameters():
                param.grad = None
                

        # del dt['video_action-label']
        
        output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)
        
        return output, loss, los
        

