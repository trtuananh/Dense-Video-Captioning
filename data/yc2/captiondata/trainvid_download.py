import json
import os

VAL_JSON = 'yc2_test.json'
DATASET_PATH = './test/'

train_file = open(VAL_JSON, 'r')
train_data = json.load(train_file)
#missing_file = open('missing.json', 'r')
#missing = json.load(missing_file)
missing = []
# 9000 - 9250, 9250 - 9500, 9500 - 9750, 9750 - 10000
print(len(train_data))
for idx, half_path in enumerate(train_data):
    
    #if idx < 400:
        #continue

    #if idx == 200:

        #break

    url = 'https://youtube.com/watch?v=' + half_path[2:]
    print(f'Current: {idx} and {half_path[2:]}')
    directory = os.path.join(DATASET_PATH, half_path)
    os.mkdir(directory)
    vid_file = os.path.join(directory, 'vid.mp4')
    sound_file = os.path.join(directory, 'sound.m4a')

    # Video download : 243
    os.system(f'yt-dlp -f 133 {url} -o {vid_file}')


    if len(os.listdir(directory)) == 0:
        os.system(f'yt-dlp -f 134 {url} -o {vid_file}')
        if len(os.listdir(directory)) == 0:
            missing.append(half_path)
            continue

    os.system(f'yt-dlp -f 250 {url} -o {sound_file}')
    
    if len(os.listdir(directory)) == 1:
        os.system(f'yt-dlp -f 139 {url} -o {sound_file}')




print(f'Missing num: {len(missing)}')