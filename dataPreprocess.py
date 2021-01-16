from cocostuffapimaster.PythonAPI.cocostuff.cocoSegmentationToPngDemo import cocoSegmentationToPngDemo
import torch
import torch.nn as nn
import torch.optim as optim
import random
from pycocotools.coco import COCO
import pycocotools.cocostuffhelper
from ConvAutoencoder import architecture, Autoencoder
import train_epoch

#cocoSegmentationToPngDemo(dataDir="cocostuffapimaster",dataTypeAnn="val2017",exportImageLimit=1e10)

cocoVal = COCO(annotation_file='cocostuffapimaster/annotations/stuff_val2017.json')

print(cocoVal.imgs)
#cocoVal.download(tarDir='cocostuffapimaster/images/val2017', imgIds=cocoVal.getImgIds())