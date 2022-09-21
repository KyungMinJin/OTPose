# OTPose

## Description

This is an official repo for OTPose: Occlusion-Aware Transformer for Pose Estimation in Sparsely-Labeled Videos. [Paper](https://arxiv.org/abs/2207.09725).

## Getting Started

1. Environment Requirement.
```terminal
conda create -n OTPose python=3.6.12
conda activate OTPose
pip install -r requirements.txt
```
[//]: # (2. Install pytorch.)

[//]: # ()
[//]: # (If you use cudatoolkit >= 11.x, you can change the cudatoolkit version. )

[//]: # (```angular2html)

[//]: # (conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -c conda-forge)

[//]: # (```)
2. Install DCN.
```angular2html
cd thirdparty/deform_conv
python setup.py develop
```

[//]: # (- trouble-shooting)

[//]: # (```angular2html)

[//]: # (subprocess.CalledProcessError: Command '['which', 'x86_64-conda_cos7-linux-gnu-c++']' returned non-zero exit status 1.)

[//]: # ($ conda install gxx_linux-64)

[//]: # (```)


### Data preparation

First, create a folder `${DATASET_DIR}`  to store the data of PoseTrack17 and PoseTrack18.

The directory structure should look like this:

```
${DATASET_DIR}
	|--${POSETRACK17_DIR}  
	|--${POSETRACK18_DIR}
	
# For example, our directory structure is as follows.
# If you don't know much about configuration file(.yaml), please refer to our settings.
DataSet
	|--PoseTrack17
	|--PoseTrack18
```

**For PoseTrack17 data**, we use a slightly modified version of the PoseTrack dataset where we rename the frames to follow `%08d` format, with first frame indexed as 1 (i.e. `00000001.jpg`). First, download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Then, rename the frames for each video as described above using [this script](https://github.com/facebookresearch/DetectAndTrack/blob/master/tools/gen_posetrack_json.py).  

Like [PoseWarper](https://github.com/facebookresearch/PoseWarper) and [DCPose](https://github.com/Pose-Group/DCPose), We provide all the required JSON files, which have already been converted to COCO format. Evaluation is performed using the official PoseTrack evaluation code, [poseval](https://github.com/leonid-pishchulin/poseval), which uses [py-motmetrics](https://github.com/cheind/py-motmetrics) internally. We also provide required MAT/JSON files that are required for the evaluation.

Your extracted PoseTrack17  directory should look like this:

```
|--${POSETRACK17_DIR}
	|--images
        |-- bonn
        `-- bonn_5sec
        `-- bonn_mpii_test_5sec
        `-- bonn_mpii_test_v2_5sec
        `-- bonn_mpii_train_5sec
        `-- bonn_mpii_train_v2_5sec
        `-- mpii
        `-- mpii_5sec
    |--images_renamed   # first frame indexed as 1  (i.e. 00000001.jpg)
     	|-- bonn
        `-- bonn_5sec
        `-- bonn_mpii_test_5sec
        `-- bonn_mpii_test_v2_5sec
        `-- bonn_mpii_train_5sec
        `-- bonn_mpii_train_v2_5sec
        `-- mpii
        `-- mpii_5sec
```

**For PoseTrack18 data**, please download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Since the video frames are named properly, you only need to extract them into a directory of your choice (no need to rename the video frames). As with PoseTrack17, we provide all required JSON files for PoseTrack18 dataset as well.

Your extracted PoseTrack18 images directory should look like this:
```
${POSETRACK18_DIR}
    |--images
        |-- test
        `-- train
        `-- val
```

### Create Symbolic link

```
ln -s  ${OTPose_SUPP_DIR}  ${OTPose_Project_Dir}  # For OTPose supplementary file
ln -s  ${DATASET_DIR}  ${OTPose_Project_Dir}      #  For Dataset


# For example
${OTPose_Project_Dir} = /your/project/path/Pose_Estimation_OTPose
${OTPose_SUPP_DIR}    = /your/supp/path/OTPose_supp_files
${DATASET_DIR}        = /your/dataset/path/DataSet

ln -s /your/supp/path/OTPose_supp_files  /your/project/path/Pose_Estimation_OTPose  # SUP File Symbolic link 
ln -s /your/dataset/path/DataSet         /your/project/path/Pose_Estimation_OTPose  # DATASET Symbolic link 2
```

### Training from scratch

**For PoseTrack17**

```
cd tools
# train 
python train.py --cfg ../configs/posetimation/OTPose/posetrack17/model_RSN.yaml 
# val
python eval.py --cfg ../configs/posetimation/OTPose/posetrack17/model_RSN.yaml 
```

The results are saved in `${OTPose_Project_Dir}/output/PE/OTPose/OTPose/PoseTrack17/{Network_structure _hyperparameters}` by default

**For PoseTrack18**

```
cd tools
# train 
python train.py --cfg ../configs/posetimation/OTPose/posetrack18/model_RSN.yaml 
# val
python eval.py --cfg ../configs/posetimation/OTPose/posetrack18/model_RSN.yaml 
```

The results are saved in `${OTPose_Project_Dir}/output/PE/OTPose/OTPose/PoseTrack18/{Network_structure _hyperparameters}` by default

### Validating/Testing from our pretrained models

We will prepare our pretrained model as soon as possible issues of our GPU server are resovled. 
```
# Evaluate on the PoseTrack17 validation set
python run.py --cfg ../configs/posetimation/OTPose/posetrack17/model_RSN_trained.yaml --val  
# Evaluate on the PoseTrack17 test set
python run.py --cfg ../configs/posetimation/OTPose/posetrack17/model_RSN_trained.yaml --test
```

### Run on video

We will prepare all visualization codes in this repo but we need to test it. We will prepare the run command as soon as possible.

[//]: # (```)

[//]: # (cd demo/                   )

[//]: # (mkdir input/)

[//]: # (# Put your video in the input directory)

[//]: # (python video.py)

[//]: # (```)


## Acknowledgements
- The code is built upon [DCPose](https://github.com/Pose-Group/DCPose).
