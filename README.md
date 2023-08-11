# YOLOv1

## Introduction

This project is based on the given drone data, utilizing VGG16 with batch normalization to train and test the YOLOv1 model, ultimately achieving optimal detection performance.

## Requirements

If you are using conda, you may configure YOLOv1 as:

```
conda creat -n yolos python=3.8.16
```

Then, use the following command to install the required packages:

```
pip install -r requirements.txt
```

If the installation fails using the previous commands, you can install the correct versions of PyTorch and CUDA separately using the following commands:

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f  https://download.pytorch.org/whl/cu111/torch_stable.htm
```

In the end, you can use either the `conda` or `pip` command to install the remaining packages.

## Dataset

The organization format of the data is as follows:

> data
>
> > train
> >
> > > video_name
> > >
> > > > 000001.jpg
> > > >
> > > > ...
> > > >
> > > > IR_label.json
> >
> > test
> >
> > ...

## Train and Test

You can use the following command for training:

```
python train.py
```

You use the following command to calculate accuracy and make predictions on the test dataset:

```
python test.py
```

## Trained model

You can download the pre-trained model from this [link]([恒源云_GPUSHARE-恒源智享云](https://gpushare.com/center/hire)).

```
Access Code：8888
```