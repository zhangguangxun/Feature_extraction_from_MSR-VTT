# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:01:29 2020

@author: Administrator
"""

import sys
sys.path.append("../detectron2")

import cv2
import io
import numpy as np
from  matplotlib import pyplot as plt
import torch
import os

from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_backbone
from detectron2.config import get_cfg

confidence_threshold = 0.5
config_file = "../detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"
model_weights = "../detectron2/demo/faster_rcnn_R_101_C4_3x.pkl"

cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS =model_weights
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
cfg.freeze()
backbone = build_backbone(cfg)
predictor = DefaultPredictor(cfg)

def getFeature(img,raw_boxes):
    """
    The input is the image and the bounding boxes;
    The output is a list which contain serveal features corresponding to the bounding boxes
    """
    raw_height, raw_width = img.shape[:2]
    image = predictor.transform_gen.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": raw_height, "width": raw_width}]
    images = predictor.model.preprocess_image(inputs)
    features = predictor.model.backbone(images.tensor)
    new_height, new_width = image.shape[:2]
    scale_x = 1. * new_width / raw_width
    scale_y = 1. * new_height / raw_height
    boxes = raw_boxes.clone()
    boxes.scale(scale_x=scale_x, scale_y=scale_y)
    proposal_boxes = [boxes]
    nfeatures = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(nfeatures, proposal_boxes)
    feature_pooled = box_features.mean(dim=[2, 3])
    del features,nfeatures,box_features,image
    return feature_pooled
    del feature_pooled
    
def fromPathGetFeature(path):
    """
    The input is the path of image;
    The output is a list of features
    """
    img = cv2.imread(path)
    pred = predictor(img)
    raw_bbxs = pred['instances'].pred_boxes
    raw_feature = getFeature(img,raw_bbxs)
    raw_feature_cpu = raw_feature.cpu()
    del pred,raw_bbxs,raw_feature
    return raw_feature_cpu

"""
I want to get the feature from each frame and store them into a list
"""
from glob import glob
paths = sorted(glob('tmp0/*.jpg'),key = lambda x: int(x[5:-4]))
finalFeatureList = []
for i, path in enumerate(paths):
    tmpFeatureList = fromPathGetFeature(path)
    finalFeatureList.append(tmpFeatureList)
    del tmpFeatureList
    print(i)
print("The total lenth of frames is",len(finalFeatureList))
