import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import timm
import segmentation_models_pytorch as smp
import imutils
from skimage.transform import ProjectiveTransform

import os
from tqdm import tqdm
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import gc
import glob

import random
import yolov5
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy
import os
from PIL import Image

# yolo_model_det = yolov5.load('/content/drive/MyDrive/final_yolo_weights.pt')
yolo_fast = yolov5.load('/content/drive/MyDrive/yolo_on_6_fast.pt')

def return_fast_output(yolo_model, img):

    image = img.copy()

    results_yolo = yolo_model(img)

    try:
        boxes = results_yolo.pred[0][:, :4].tolist()
        scores = results_yolo.pred[0][:, 4].tolist()
        labels = results_yolo.pred[0][:, 5].tolist()
    except:
        boxes = []
        scores_yolo = []
        labels_yolo = []

    dic = {}
    for each in labels:
        if each not in dic.keys():
            dic[each] = (0,[])

    for i in range(len(labels)):
        score , box = dic[labels[i]]
        if score < scores[i]:
            dic[labels[i]] = (scores[i], boxes[i])

    # print(dic)
    return dic
     # Load the transformed image
transformed_image = Image.open(transformed_image_path)

                # Perform detection and inference
transformed_img=np.array(transformed_image)
# detection_dict = number_detection(transformed_img, mode)
temp = return_fast_output(yolo_fast, transformed_img)
  
