import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from optical_flow import *
TRACKER_POINTS = 500 # How many points will be used to track the optical flow
CRAZY_LINE_DISTANCE =  50 # Distance value to detect crazy lines
CRAZY_LINE_LIMIT = 100 * TRACKER_POINTS / 1000 # Amount of crazy lines are indication of different shots
ABSDIFF_ANGLE = 20 # To determine the inconsistency between tangent values in degrees
LINE_THICKNESS = 3 # Lines thickness that we will use for mask delta
CONTOUR_LIMIT = 10 # Contour limit for detecting ZOOM, ZOOM + PAN, ZOOM + TILT, ZOOM + ROLL (Not just PAN, TILT, ROLL)
TARGET_HEIGHT = 360 # Number of horizontal lines for target video and processing. Like 720p, 360p etc.
DELTA_LIMIT_DIVISOR = 3 # Divisor for detecting too much motion. Like: ( height * width / X )

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]
F = 1
bbox = np.zeros((F,2,2))
if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    c, r, w, h  = detect_one_face(frame)
    

    for f in range(F):
        bbox[f] = np.array([(c,r), (c+w, r+h)])

    # Parameters for lucas kanade optical flow
    lk_params_winSize=(15, 15)
    lk_params_maxLevel=2
    lk_params_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    ret, old_frame = ret, frame
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    
    # num = 8
    # for i in range(num - 1):
    #     p0_temp = np.random.rand(1,2)*(w/2,h/2) + (c+w/4,r+h/4)
    #     features.append(p0_temp)

    # p0_temp = [[c+w/2, r+h/2]]
    # features.append(p0_temp)
    # features = np.array(features).astype('float32')

    features = []
    features = getFeatures(old_gray, bbox)
    new_features = []
    
    #generating random colors to apply to the resulting 2D vectors
    resultingvectorcolor = np.random.randint(0, 255, (100, 3))
    #mask image is created for drawing the vectors
    resmask = np.zeros_like(frame)

    while(1):
        ret, frame = video.read()
        # show the output frame
        if ret == False:
            break

       
        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        for i in range(bbox.shape[0]):
            feature = features[i,:,:] 
            print("debugg:\n",feature)
            new_feature, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                    frame_gray,
                                                    feature,
                                                    None,
                                                    maxLevel = lk_params_maxLevel,
                                                    winSize = lk_params_winSize,
                                                    criteria = lk_params_criteria
            )
             
            # Select good points
            good_new = new_feature[st == 1]
            good_old = feature[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                resmask = cv2.line(resmask, (int(a), int(b)), (int(c), int(d)),resultingvectorcolor[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5,resultingvectorcolor[i].tolist(), -1)
                frame = cv2.add(frame, resmask)
            # update new features
            new_feature = good_new.reshape(-1, 1, 2)
            new_features[i] = new_feature
       
       
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()

        # detec object
        c, r, w, h = detect_one_face(frame)

        # Use face detector else use optical flow
        if w != 0 and h != 0:
            pos = (c + w/2, r + h/2)
            for f in range(F):
                bbox[f] = np.array([(c,r), (c+w, r+h)])
            features = getFeatures(old_gray, bbox)
        else:
            pass
            # pos = sum(list(zip(*features))[0]) / num
            # print(pos)
            # c = int((pos[0]-c)*2)
            # r = int((pos[1]-h)*2)
        
        cv2.rectangle(frame, (c, r), (c+w, r+h), (0, 255, 0), thickness=2)

        bbox = applyGeometricTransformation(features, new_features, bbox, frame.shape)
        # new_features = features
        # display the bbox
        for f in range(F):
            cv2.rectangle(frame, tuple(bbox[f,0].astype(np.int32)), tuple(bbox[f,1].astype(np.int32)), (0,0,255), thickness=2)
     
        print("Debugg2:",type(bbox))
        print(bbox)
        print(bbox.shape)


        # update features
        features =new_features
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()