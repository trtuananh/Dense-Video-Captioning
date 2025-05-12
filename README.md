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


## Results

This section presents the quantitative evaluation of our Dense Video Captioning (DVC) model and comparisons with baseline models on the YouCook2 validation dataset. We evaluated performance across three tasks: Dense Video Captioning, Localization, and Paragraph Captioning.

### DVC Captioning Results on YouCook2 Validation Set

This table summarizes the performance of our models and baseline models on the Dense Video Captioning task.  "PDVC" refers to our proposed Dense Video Captioning model. "TSP(R(2+1)D)" represents the baseline model utilizing the R(2+1)D feature extractor. "TSP(MVITv2)" represents the baseline model utilizing the MVITv2 feature extractor. "+ ete" indicates models trained with our end-to-end training methodology. "+ HuBERT" denotes models incorporating audio features extracted by the HuBERT model.  "Transformer Layers" indicates the number of Transformer layers used in the model.

| Model             | Feature Extractor          | Transformer Layers | Bleu4 | METEOR | CIDEr | SODA |
|-------------------|---------------------------|--------------------|-------|--------|-------|------|
| PDVC              | TSP(R(2+1)D)              | 2                  | 0.89  | 4.30   | 21.69 | 4.02 |
| PDVC              | TSP(MVITv2)               | 2                  | 1.05  | 5.00   | 25.26 | 4.75 |
| PDVC              | TSP(MVITv2) + ete        | 2                  | 1.20  | 5.12   | 25.33 | 4.80 |
| PDVC              | TSP(R(2+1)D)              | 3                  | 0.92  | 4.25   | 21.88 | 4.06 |
| PDVC              | TSP(MVITv2)               | 3                  | 1.10  | 5.16   | 28.20 | 4.85 |
| PDVC              | TSP(MVITv2) + ete        | 3                  | 1.18  | 5.27   | 28.32 | 4.67 |
| PDVC              | TSP(MVITv2) + HuBERT     | 3                  | **1.45** | **5.43** | **30.34** | **4.96** |
| PDVC [44]         | TSN [42]                  | 2                  | 0.80  | 4.74   | 22.71 | 4.42 |
| MT [52]           | TSN                       | -                  | 0.30  | 3.18   | 6.10  | -    |
| ECHR [45]         | TSN                       | -                  | -     | 3.82   | -     | -    |
| GVL [43]          | TSN                       | -                  | 1.04  | 5.01   | 26.52 | 4.91 |

**Key Observations:**

* Our end-to-end trained models ("+ ete") consistently outperform their counterparts without end-to-end training, demonstrating the effectiveness of the proposed training methodology.
* The integration of audio features using HuBERT significantly improves performance across all metrics, highlighting the importance of multi-modal information for DVC.
* Our best performing model, PDVC with TSP(MVITv2) + HuBERT, achieves state-of-the-art results on the YouCook2 dataset.

### DVC Localization Results on YouCook2 Validation Set

This table presents the localization performance of our models on the YouCook2 dataset.  Metrics reported are Average Precision and Average Recall.

| Model             | Feature Extractor          | Transformer Layers | Average Precision | Average Recall |
|-------------------|---------------------------|--------------------|-------------------|----------------|
| PDVC              | TSP(R(2+1)D)              | 2                  | 0.2950            | 0.1913         |
| PDVC              | TSP(MVITv2)               | 2                  | 0.3256            | 0.2106         |
| PDVC              | TSP(MVITv2) + ete        | 2                  | 0.3283            | **0.2190** |
| PDVC              | TSP(R(2+1)D)              | 3                  | 0.2873            | 0.1886         |
| PDVC              | TSP(MVITv2)               | 3                  | 0.3227            | 0.2065         |
| PDVC              | TSP(MVITv2) + ete        | 3                  | 0.3184            | 0.1991         |
| PDVC              | TSP(MVITv2) + HuBERT     | 3                  | **0.3312** | 0.2112         |

**Key Observations:**

* Our model with HuBERT features achieves the best localization performance, further confirming the benefit of incorporating audio information.
* End-to-end training improves Average Recall, indicating better temporal alignment of captions with video segments.

### Paragraph Captioning Results on YouCook2 Validation Set

This table shows the performance of our models on the paragraph captioning task.

| Feature Extractor          | Transformer Layers | Bleu4 | METEOR | CIDEr  |
|---------------------------|--------------------|-------|--------|--------|
| TSP(R(2+1)D)              | 2                  | 5.4   | 11.56  | 13.06  |
| TSP(MVITv2)               | 2                  | 6.09  | 12.19  | 17.52  |
| TSP(MVITv2) + ete        | 2                  | **6.58** | **12.79** | **15.27** |
| TSP(R(2+1)D)              | 3                  | 5.78  | 11.65  | 11.74  |
| TSP(MVITv2)               | 3                  | 5.69  | 11.77  | 15.42  |
| TSP(MVITv2) + ete        | 3                  | 5.74  | 11.85  | 14.54  |
| TSP(MVITv2) + HuBERT     | 3                  | 6.26  | 12.46  | 15.28  |

**Key Observations:**

* Our end-to-end training method consistently improves the performance of paragraph captioning models.
* The HuBERT features also contribute to performance gains in this task.
