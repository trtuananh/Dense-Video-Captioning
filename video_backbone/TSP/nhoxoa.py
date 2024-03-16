from torchvision.io.video import read_video
import torch
from torch import nn
from torchvision.transforms._presets import VideoClassification
from functools import partial
from torchvision.models.video import r3d_18, R3D_18_Weights, mvit_v2_s, MViT_V2_S_Weights
from models.model import Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vid =  torch.randint(0, 256, (16, 3, 400, 400)) # T C H W
vid = vid[:32]  # optionally shorten duration

# Step 1: Initialize model with the best available weights
weights = MViT_V2_S_Weights.DEFAULT
model = mvit_v2_s(weights=weights)
#model.head = nn.Sequential()
model.eval()

print(model)

preprocess = weights.transforms()
input = preprocess(vid).unsqueeze(0)
print(input.shape)
fea = model(input)
print(fea.shape)

# model = Model('mvitv2', [110, 2], 2, True)

# for i, layer in enumerate(model.head):
#     if i == 1:
#         print(layer.in_features)

# print(count_parameters(model))

# r2plus1d_34: 63550455
# mvitv2: 34 317 296 
