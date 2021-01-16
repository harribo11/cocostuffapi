import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocostuffhelper import cocoSegmentationToSegmentationMap, segmentationToCocoResult
from pycocotools import coco
from PIL import Image
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from ConvAutoencoder import architecture, Autoencoder
from train_epoch import train_epoch


class CocoStuff(dset.VisionDataset):
    def __init__(
            self,
            root: str,
            annFile: str,
            transform= None,
            target_transform=None,
            transforms=None,):
        super(CocoStuff, self).__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annotation_file=annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def __getitem__(self, index:int ):
        coco = self.coco
        img_id = self.ids[index]
        
        mask = cocoSegmentationToSegmentationMap(coco, img_id)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        
        return img, mask, img_id
    
    def __len__(self):
        return len(self.ids)

class PadImages:
    def __init__(self, batch, padToMultiple=16):
        height = 0
        width = 0
        for img, mask, imgId in batch:
            h = img.shape[1]
            w = img.shape[2]
            if h > height:
                height = h
            if w > width:
                width = w

        if height % padToMultiple != 0:
          height += padToMultiple - (height % padToMultiple)
        
        if width % padToMultiple != 0:
          width += padToMultiple - (width % padToMultiple)

        imgs = []
        masks = []
        paddings = []
        imgIds = []

        for img, mask, imgId in batch:
            h = img.shape[1]
            w = img.shape[2]
            padTop = (height - h) // 2
            padLeft = (width - w) // 2
            #in case of odd numbers
            padBottom = (height - h) - padTop
            padRight = (width - w) - padLeft

            padding = (padLeft, padRight, padTop, padBottom) #used to unpad later
            paddings.append(padding)

            imgIds.append(imgId)

            pad = nn.ZeroPad2d(padding=padding)
            imgs.append(pad(torch.as_tensor(img)))
            masks.append(pad(torch.as_tensor(mask)))
            
        
        self.imgs = torch.stack(imgs)
        self.masks = torch.stack(masks)
        self.paddings = paddings
        self.imgIds = imgIds

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.imgs = self.imgs.pin_memory()
        self.masks = self.masks.pin_memory()
        return self.imgs, self.masks, self.paddings, self.imgIds

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def showSegmentationGroundTruth(cocoData):
    imgIds = cocoData.getImgIds()
    imgId = imgIds[np.random.randint(0, len(imgIds))]
    print('Processing image', imgId, '\n' )
    img = cocoData.loadImgs(imgId)[0]

    # Load and display image
    I = skimage.io.imread(img['coco_url'])
    plt.figure()
    plt.subplot(121)
    plt.imshow(I)
    plt.axis('off')
    plt.title('original image')

    # Load and display stuff annotations
    annIds = cocoData.getAnnIds(imgIds=img['id'])
    anns = cocoData.loadAnns(annIds)
    plt.subplot(122)
    plt.imshow(I)
    cocoData.showAnns(anns)
    plt.axis('off')
    plt.title('annotated image')
    plt.show()

def showSegmentationModelGenerated(model, data, dataLoader, numToShow=10):
  cocoData = coco.COCO(annotation_file='cocostuffapimaster/annotations/stuff_val2017.json')
  categories = cocoData.loadCats(cocoData.getCatIds())

  n = 0

  for (idx, batch) in enumerate(dataLoader):
    plt.figure()
    batch_x = batch[0]
    batch_y = batch[1].to(torch.int64)
    pad = batch[2]
    imgId = batch[3][0]

    with torch.no_grad():
      outputs = model.forward(batch_x.cuda()).cpu()
    del batch_x
    del batch_y

    img = cocoData.loadImgs(imgId)[0]

    I = skimage.io.imread(img['coco_url'])
    plt.subplot(121)
    plt.imshow(I)
    plt.axis('off')
    plt.title('original image')
    # Load and display stuff annotations
    mask = torch.argmax(outputs[0],0)


    idCount = torch.bincount(mask.reshape(-1), weights=None)
    cats = {}
    for category in categories:
      if idCount[category['id']] != 0:
        cats[category['name']] = idCount[category['id']].item()
    print(cats)
    anns = segmentationToCocoResult(mask, imgId)
    plt.subplot(122)
    plt.imshow(I)
    cocoData.showAnns(anns)
    plt.axis('off')
    plt.title('annotated image')    
    plt.show()
    n += 1
    if n == numToShow:
      break
  





