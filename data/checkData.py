import os
import json

file = open('./anet/captiondata/train_modified.json', 'r')
anet_data = json.load(file)


maxLen = -1
for key in anet_data:
    if len(anet_data[key]['timestamps']) > maxLen:
        maxLen = len(anet_data[key]['timestamps']) 

print(maxLen)