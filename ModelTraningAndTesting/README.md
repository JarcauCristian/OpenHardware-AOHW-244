# Description

This project creates classification models which are able to discriminate between attacked 
and genuine files. The image files are attacked with various algorithms as found in [Foolbox: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX](https://pypi.org/project/foolbox/).

## Setup

Run `pip install -r requirements.txt` from the project.

## Data download and generation of attacked files

The dataset used in this project is GTSRB - German Traffic Sign Recognition Benchmark. 

As the data files are too large, they are not included here. Thy can be downloaded and further processed locally. 

Inside 'iesc_adversarial/data':

1. download the zip file from [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). Unzip it, it should appear as iesc_adversarial/data/german/archive and it should contain: `meta  Meta  test  Test  Test.csv  train  Train  Train.csv`
2. run script `iesc_adversarial/code/attack_images_step1.py` which produces inside `data/german` the following dirs: `DATA_224x224` and `split`. Note that 224x224 is the size to which GTSRB images are resized. Resizing keeps aspect ratio via letterboxing. 
3. run script `detector/generate_train_test_splits_attacked_step2.py` to produce a train/test split. Each of the train/test subsets contains classes (subdirs):  `0=genuine` and `1=attacked`.

## Train the classifier

The classifier is based on resnet family. Currently the resnet50 is used as a starting point for transfer learning and fine tuning. Training and test of the classifier is done by `detector/attack_detector/detect_attack_cifar_step3.py`. 

I highly recommend you to keep looking to the tensorboard graph, which plots loss and accuracy realtime in directories `runs`. Run tensorboard with `tensorboard --logir=runs` inside `attack_detector` directory. 

## Test the accuracy of the classifier

- For the unoptimized version run: `python unoptimized.py`, this will load the best model trained and will apply the model on unseend images to compute the avarage accuracy over 10 runs, as well the avarage runtime.

- For the optimized version run: `python optimized.py`, this will load the best model trained and will apply the model on unseend images to compute the avarage accuracy over 10 runs, as well the avarage runtime.
