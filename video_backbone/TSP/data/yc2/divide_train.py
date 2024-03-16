import os
import csv
import json
import pandas as pd

JSON_YC2 = './youcookii_annotations_trainval.json'
LABEL_CSV = './label_foodtype.csv'
TRAIN_METADATA = './yc2_train_metadata.csv'
VALID_METADATA = './yc2_valid_metadata.csv'

json_file = open(JSON_YC2, 'r')
yc2_data = json.load(json_file)

label_file = open(LABEL_CSV, 'r')
csv_reader_label = csv.reader(label_file, delimiter=',')

metadata_train = pd.read_csv(TRAIN_METADATA, header=0) 
metadata_valid = pd.read_csv(VALID_METADATA, header=0)

idx2Food = {}

for line in csv_reader_label:
    idx2Food[str(line[0])] = line[1]    

header = ['video-name', 't-start', 't-end', 'action-label', 
          'video-duration', 'filename','fps', 'video-frames', 
          'temporal-region-label'
        ]


def create_row(
        video_name, 
        t_start, 
        t_end, 
        action_label, 
        video_duration, 
        filename, 
        fps, 
        video_frames, 
        temporal_region_label
    ):
    row = []
    row.append(video_name)
    row.append(t_start)
    row.append(t_end)
    row.append(action_label)
    row.append(video_duration)
    row.append(filename)
    row.append(fps)
    row.append(video_frames)
    row.append(temporal_region_label)
    return row

training_rows = []
valid_rows = []

def create_ground_truth(type='foreground'):

    for video_name in yc2_data['database']:
        dataFrame = metadata_train
        if yc2_data['database'][video_name]['subset'] == 'validation':
            dataFrame = metadata_valid

        vid_row = dataFrame.loc[dataFrame['filename']== f'v_{video_name}.mp4']


        annotations = yc2_data['database'][video_name]['annotations']
        duration = float(vid_row['video-duration']) if not vid_row.index.empty else None

        if duration is None:
            continue

        segments = []
        
        if type == 'background':
            cumulate = 0

            for segment_id_sentence_block in annotations:
                t_start = segment_id_sentence_block['segment'][0]
                t_end = segment_id_sentence_block['segment'][1]
                if t_start - 1 - cumulate >= 8:
                    segments.append([cumulate, t_start - 1])
                cumulate = t_end + 1

            if duration - 1 - cumulate >= 8:
                segments.append([cumulate, duration])

        else:
            for segment_id_sentence_block in annotations:
                t_start = segment_id_sentence_block['segment'][0]
                t_end = segment_id_sentence_block['segment'][1]
                segments.append([t_start, t_end])


        
        num_frames = int(vid_row['video-frames']) if not vid_row.index.empty else None
        
        for segment in segments:
            row = create_row(
                video_name, segment[0],
                segment[1],
                idx2Food[yc2_data['database'][video_name]['recipe_type']] if type=='foreground' else '',
                duration,
                f'v_{video_name}.mp4',
                30, 
                num_frames,
                'Action' if type=='foreground' else 'No action'
            )

            if yc2_data['database'][video_name]['subset'] == 'training':
                training_rows.append(row)
            else:
                valid_rows.append(row)


# Create ground truth for forground-action segments

# Create ground truths for backround-action segments

create_ground_truth()
create_ground_truth(type='background')


with open('./yc2_train_tsp_groundtruth.csv', 'w') as gt_train_file:
    csv_writer_train = csv.writer(gt_train_file)
    csv_writer_train.writerow(header)
    csv_writer_train.writerows(training_rows)


with open('./y2c_valid_tsp_groundtruth.csv', 'w') as gt_valid_file:
    csv_writer_valid = csv.writer(gt_valid_file)
    csv_writer_valid.writerow(header)
    csv_writer_valid.writerows(valid_rows)