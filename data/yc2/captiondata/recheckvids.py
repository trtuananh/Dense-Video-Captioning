import json
import os
'''
    Missing test 15
    Justone test 0

    Missing valid 42
    Justone valid 0

    Missing train 146

    Con lai 1797 vids
'''
VAL_JSON = 'yc2_val.json'
DATASET_PATH = '/home/nam/Game-drive/yc2_gvf_npy'

train_file = open(VAL_JSON, 'r')
train_data = json.load(train_file)
#missing_file = open('missing.json', 'r')
#missing = json.load(missing_file)
missing = []
# 9000 - 9250, 9250 - 9500, 9500 - 9750, 9750 - 10000

newdata = {}

for video_name in train_data:
    npy_name = video_name + '.npy'
    if npy_name in os.listdir(DATASET_PATH):
        newdata[npy_name] = train_data[video_name]
    else:
        missing.append(video_name)

with open('yc2_newval.json', 'w') as file:
    json.dump(newdata, file)

print(len(missing))