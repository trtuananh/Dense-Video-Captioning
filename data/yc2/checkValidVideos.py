import os
import json


FEATURE_FILE = './features/tsp'

train_file = open('./captiondata/yc2_train.json', 'r')
train_data = json.load(train_file)

invalids = []
count = 0
c = 0

for videoName in train_data:
    count += 1
    npyFile = videoName + '.npy'
    if npyFile not in os.listdir(FEATURE_FILE):
        invalids.append(videoName)
        c += 1

val_file = open('./captiondata/yc2_val.json', 'r')
val_data = json.load(val_file)

for videoName in val_data:
    count += 1
    npyFile = videoName + '.npy'
    if npyFile not in os.listdir(FEATURE_FILE):
        invalids.append(videoName)
        c += 1


test_file = open('./captiondata/yc2_test.json', 'r')
test_data = json.load(test_file)

for videoName in test_data:
    count += 1
    npyFile = videoName + '.npy'
    if npyFile not in os.listdir(FEATURE_FILE):
        invalids.append(videoName)
        c += 1

print(f'Total: {count}')        
print(f'Missing {c} video')

with open('./features/invalidVids.json', 'w') as out:
    json.dump(invalids, out)