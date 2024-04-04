import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from optical_flow import *
import copy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Parameters for lucas kanade optical flow
lk_params_winSize=(15, 15)
lk_params_maxLevel=2
lk_params_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]
# Define how many objects to track
F = 1
frame_cnt = 0 
#generating random colors to apply to the resulting 2D vectors
resultingvectorcolor = np.random.randint(0, 255, (100, 3))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    #init old_frame
    ret, frame = video.read()
    ret, frame_old = ret, frame

    # convert to gray scale
    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

    #detect face
    x, y, w, h  = detect_one_face(frame)

    bbox = np.zeros((F,2,2))

    for f in range(F):
        bbox[f] = np.array([(x,y), (x+w, y+h)])

    # features = cv2.goodFeaturesToTrack(gray_old, mask = None, **feature_params)
    features = getFeatures(gray_old, bbox)
    print("Debugg1:",type(features))
    print(features)
    print(features.shape)
    resmask = np.zeros_like(frame)
    while(1):
        
        ret, frame = video.read()
   
        
        # get feature 
        vis = copy.deepcopy(frame)

        # convert to Grayscale
        gray = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

        # if w != 0 and h != 0:
        #     bbox = np.zeros((F,2,2))
        
        #     # Manually select objects on the first frame
        #     for f in range(F):
        #         # cv2.destroyAllWindows()
        #         bbox[f] = np.array([(x,y), (x+w, y+h)])
        #         gray_tempt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        #     features = getFeatures(gray_tempt, bbox)

        # calculate optical flow
        newfeatures, st, err = cv2.calcOpticalFlowPyrLK(gray_old,
                                               gray,
                                               features,
                                               None,
                                               maxLevel = lk_params_maxLevel,
                                               winSize = lk_params_winSize,
                                               criteria = lk_params_criteria
        )

        # Select good points
        good_new = newfeatures[st == 1]
        good_old = features[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            resmask = cv2.line(resmask, (int(a), int(b)), (int(c), int(d)),resultingvectorcolor[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5,resultingvectorcolor[i].tolist(), -1)
            frame = cv2.add(frame, resmask)

        #update feature
        
        features = good_new.reshape(-1, 1, 2)
        print(features)

        # save frame_old
        gray_old = gray.copy()

        x, y, w, h  = detect_one_face(frame)

        # display feature points
        for feature in features[f]:
            cv2.circle(vis, tuple(feature.astype(np.int32)), 2, (0,255,0), thickness=-1)
        # for f in range(F):
            # for feature in features[f]:
            #     cv2.circle(vis, tuple(feature.astype(np.int32)), 2, (0,255,0), thickness=-1)
        # display bounding box of the face
        
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        
        # show the output frame
        if ret == False:
            break
        
        cv2.imshow('frame',vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()