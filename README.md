# THESIS PROJECT

[[paper]](https://arxiv.org/abs/2108.07781) 
**This repo supports:**
* two video captioning tasks: dense video captioning and video paragraph captioning
* two datasets: ActivityNet Captions and YouCook2
* video features containing C3D, TSN, TSP and MVITv2.
* visualization of the generated captions of your own videos

**Table of Contents:**
* [Introduction](#introduction)
* [Preparation](#preparation)
* [Training and Validation](#training-and-validation)
* [Running PDVC on Your Own Videos](#running-pdvc-on-your-own-videos)


## Introduction
PDVC is a simple yet effective framework for end-to-end dense video captioning with parallel decoding (PDVC), by formulating the dense caption generation as a set prediction task. Without bells and whistles, extensive experiments on ActivityNet Captions and YouCook2 show that PDVC is capable of producing high-quality captioning results, surpassing the state-of-the-art methods when its localization accuracy is on par with them.
![pdvc.jpg](pdvc.jpg)

## Preparation
Environment: Linux,  GCC>=5.4, CUDA == 11.7, Python>=3.7, PyTorch>=2.0.0

1. Clone the repo
```bash
git clone --recursive https://github.com/trtuananh/Dense-Video-Captioning.git
```

2. Create virtual environment by conda
```bash
conda create -n PDVC python=3.9
source activate PDVC
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7.0 -c pytorch
conda install ffmpeg
pip install -r requirement.txt
```

3. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd pdvc/ops
sh make.sh
```
4. Download Video Features

Download tsp_mvitv2 visual feature folder (https://drive.google.com/drive/folders/1495BandMtWAwWpJtDsds-CgvpWWs9sgb?usp=sharing) and put it in data/yc2/features.
Download sound_feature_train sound feature folder (https://drive.google.com/drive/folders/1xXC7Z4iL3HvDNtxwSYfPGo6LKgxC3mYE?usp=sharing) and put it in data/yc2/features

5. Download checkpoints

Download PDVC checkpoint trained on YouCook2 (https://drive.google.com/drive/folders/13EeAF3DPpAIpBieQ1vwF-kcCsR8bgSt4?usp=sharing) and put it in save/ to train from scratch.
Download the whole model for inference (https://drive.google.com/drive/folders/1lhTb31tgIWZUUtIO6Jt1yOjaIGr6fdoc?usp=sharing) and put it in save/.


## Training and validation
```bash
GPU_ID=0
config_path=cfgs/yc2_newModel_sound.yml
python newTrain.py --cfg_path ${config_path} --gpu_id ${GPU_ID}

# Evaluation
eval_caption_file=data/yc2/captiondata/yc2_val.json
eval_folder=yc2_newModel_sound # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID} --eval_caption_file ${eval_caption_file}
```




## Running PDVC on Your Own Videos
```bash
video_folder=visualization/videos
output_folder=visualization/output
pdvc_model_path=save/yc2_newModel_sound_best_q/model-best.pth
output_language=en
bash test_and_visualize.sh $video_folder $output_folder $pdvc_model_path $output_language
```
check the `$output_folder`, you will see a new video with embedded captions. 
Note that we generate non-English captions by translating the English captions by GoogleTranslate. 
To produce Chinese captions, set `output_language=zh-cn`. 
For other language support, find the abbreviation of your language at this [url](https://github.com/lushan88a/google_trans_new/blob/main/constant.py), and you also may need to download a font supporting your language and put it into `./visualization`.

<!-- ![demo.gif](visualization/xukun_en.gif)![demo.gif](visualization/xukun_cn.gif) -->


## Google Colab training 
We also support training with Google Colab Notebook. (https://colab.research.google.com/drive/1Kp9vHPRoG-gtL6-iHhu_CJoEc-TZJ15Q?usp=sharing)
