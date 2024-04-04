import numpy as np
import cv2
from skimage import transform as tf
import matplotlib.pyplot as plt

from helpers import *

def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (F, N, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    if len(bbox.shape) == 2:
        bbox = bbox.reshape(1,bbox.shape[0],bbox.shape[1])
    bbox = bbox.astype(int)

    features = np.zeros((bbox.shape[0],25,2))
    # features = np.zeros((25,1,2))
    for i in range(bbox.shape[0]):
        bbox_img = img[bbox[i,0,1] : bbox[i,1,1]+1 , bbox[i,0,0] : bbox[i,1,0]+1]
        corners = cv2.goodFeaturesToTrack(bbox_img,25,0.01,5)
        #corners = np.int32(corners)


        # if corners is None:
        #     corners = np.zeros((25,1,2))
        #     continue
        # corners[:,0,0] = corners[:,0,0] + bbox[i,0,0]
        # corners[:,0,1] = corners[:,0,1] + bbox[i,0,1]

    # return corners
        if corners is None:
            features[i] = 0
            continue
        x = corners[:,0,0] + bbox[i,0,0]
        y = corners[:,0,1] + bbox[i,0,1]
        # features[i,:,0] = x
        # features[i,:,1] = y
        features[i,:corners.shape[0],0] = x
        features[i,:corners.shape[0],1] = y
    return features

def applyGeometricTransformation(features, new_features, bbox, img2_sh):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in secon frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    print(features[:,0,:])
    for i in range(features.shape[1]):
        
        non_zero_entries    = ~np.logical_and( (features[:,i,0]==0) ,(features[:,i,1] == 0) )
        # print("Debugg2:",type(non_zero_entries))
        # print(non_zero_entries)
        # print(non_zero_entries.shape)
        features_p          = features[:,0,:]
        new_features_p      = new_features[:,0,:]
        tform               = tf.estimate_transform('similarity',features_p,new_features_p)
        
        bbox[i] = tform(bbox[i])

    return bbox