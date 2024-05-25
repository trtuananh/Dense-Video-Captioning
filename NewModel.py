import torch
import numpy as np
import math
import os
import torchaudio
from torch import nn
from pdvc.pdvc import build

class NewModel(nn.Module):

    def __init__(self, backbone, num_classes, num_heads, args, concat_gvf, device, transforms_train=None, transforms_valid=None):

        super(NewModel, self).__init__()
        # self.tspModel = Model(backbone=backbone, num_classes=num_classes, num_heads=num_heads, concat_gvf=concat_gvf)
        self.pdvcModel, self.pdvcCriterion, self.pdvcPostprocessor = build(args)
        self.args = args
        self.device = device
        self.transforms_train = transforms_train
        self.transforms_valid = transforms_valid
        
        self.ln1 = nn.LayerNorm(768)
        self.mha1 = nn.MultiheadAttention(768, 32, batch_first=True)
        self.mlp_seq1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )
        
        self.ln2 = nn.LayerNorm(768)
        self.mha2 = nn.MultiheadAttention(768, 32, batch_first=True)
        self.mlp_seq2 = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )
        
        self.bundle = torchaudio.pipelines.HUBERT_BASE
        self.sound_model = self.bundle.get_model()
        self.sound_model.requires_grad_ = False
        
        
    
    def visual_self_attention(self, clips):     # (1, T, 768)
        add = clips
        final_feature, _ = self.mha1.forward(query=clips, key=clips, value=clips)        # (1, T, 768)
        final_feature = self.ln1(final_feature)  # (1, T, 768)
        final_feature = final_feature + add     # (1, T, 768)
        
        
        add = final_feature
        final_feature = self.mlp_seq1(final_feature)
        final_feature = final_feature + add
        return final_feature
    
    
    
    def visual_sound_attention(self, clips, sound_feature):     # (1, T, 768)
        add = clips
        final_feature, _ = self.mha2.forward(query=sound_feature, key=clips, value=clips)        # (1, T, 768)
        final_feature = self.ln2(final_feature)  # (1, T, 768)
        final_feature = final_feature + add     # (1, T, 768)
        
        
        add = final_feature
        final_feature = self.mlp_seq2(final_feature)
        final_feature = final_feature + add
        return final_feature

    def forward(self, x, alphas=None, eval_mode=False, visualization='no'):

        dt = x
        del dt['video_action-label']
        del dt['video_temporal-region-label']
        del dt['video_gvf']
        
        x = dt['video_segment']     # [(start, end), ...]
            
        T = len(x)
        filename = dt['video_filename']
        los = 0
        clips = self.get_vid_features(filename, visualization=visualization).to(self.device)      # (T, 768)
        
        
        sound_feature = self.get_mfcc(x, filename, visualization=visualization).to(self.device)  # (T, 768)
  
        final_feature = self.visual_self_attention(clips.unsqueeze(0))  # (1, T, 768)
        final_feature = self.visual_sound_attention(final_feature, sound_feature.unsqueeze(0))
        
       
    
        dt['video_tensor'] = final_feature          # (1, T, 768)
        
        del dt['video_segment']
        
        output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)
        
        return output, loss, los
        


    def get_mfcc(self, segments, filename, visualization):
        '''
            Lay ra video_frames va mfcc
            segments: list of tuples (clip_t_start, clip_t_end)
            filename: ten video file
            eval_mode: True if dang validation
        '''
        if visualization == 'no' and os.path.exists(f'data/yc2/features/sound_feature_train/{filename[-17:-4]}.pth'):
            sound_feature = torch.load(f'data/yc2/features/sound_feature_train/{filename[-17:-4]}.pth')
            return sound_feature
        
        lst_audio = []
        try:
            waveform, sr = torchaudio.load(filename)
        except:
            return torch.zeros((len(segments), 768))

        for clip_t_start, clip_t_end in segments:
            
            start_sample = math.floor(clip_t_start * sr)
            end_sample = math.floor(clip_t_end * sr)
            cut_waveform = waveform[:, start_sample:end_sample]
            cut_waveform = torch.mean(cut_waveform, dim=0).unsqueeze(0)     
            cut_waveform = torchaudio.functional.resample(cut_waveform, sr, self.bundle.sample_rate).to(self.device)    # (1, x)
            
            with torch.no_grad():
                sound_feature, _ = self.sound_model.extract_features(cut_waveform)    # (1, 53 +- 1, 768) 
                
            sound_feature = torch.mean(sound_feature[-1], dim=1).squeeze()    # (768)
      
            
            
            lst_audio.append(sound_feature)
        
        sound_feature = torch.stack(lst_audio)
        
        if visualization == 'no':
            torch.save(sound_feature, f'data/yc2/features/sound_feature_train/{filename[-17:-4]}.pth')

        return sound_feature         # (T, 768)
    

    def get_vid_features(self, filename, visualization):
        filename = os.path.join('data/yc2/features/tsp_mvitv2', filename[-17:-4] + '.npy')
        if visualization == 'yes':
            filename = f'visualization/output/video_backbone/TSP/checkpoints/mvit_tsp.pth_stride_16/{filename[-17:-4]}.npy'
        vid_features = np.load(filename)
        vid_features = torch.from_numpy(vid_features)     # (T, 768)

        return vid_features     # (T, 768)
