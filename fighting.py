import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import numpy as np
from skimage import transform as transform

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
"""
Using Kalman Filter as a point stabilizer to stabiliz a 2D point.
"""
class Stabilizer:
    """Using Kalman filter as a point stabilizer."""
    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        """Initialization"""
        # Currently we only support scalar and point, so check user input first.
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

        # Store the parameters.
        self.state_num = state_num
        self.measure_num = measure_num

        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Store the state.
        self.state = np.zeros((state_num, 1), dtype=np.float32)

        # Store the measurement result.
        self.measurement = np.array((measure_num, 1), np.float32)

        # Store the prediction.
        self.prediction = np.zeros((state_num, 1), np.float32)

        # Kalman parameters setup for scalar.
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure

        # Kalman parameters setup for point.
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        """Update the filter"""
        # Make kalman prediction
        self.prediction = self.filter.predict()

        # Get new measurement
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])

        # Correct according to mesurement
        self.filter.correct(self.measurement)

        # Update state value.
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """Set new value for processNoiseCov and measurementNoiseCov."""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure
class Tracker:
    """Lucas-Kanade sparse optical flow tracker"""

    def __init__(self):
        self.track_len = 5
        self.tracks = []
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def update_tracks(self, img_old, img_new):
        """Update tracks."""
        # Get old points, using the latest one.
        points_old = np.float32([track[-1]
                                 for track in self.tracks]).reshape(-1, 1, 2)

        # Get new points from old points.
        points_new, _st, _err = cv2.calcOpticalFlowPyrLK(
            img_old, img_new, points_old, None, **self.lk_params)

        # Get inferred old points from new points.
        points_old_inferred, _st, _err = cv2.calcOpticalFlowPyrLK(
            img_new, img_old, points_new, None, **self.lk_params)

        # Compare between old points and inferred old points
        error_term = abs(
            points_old - points_old_inferred).reshape(-1, 2).max(-1)
        point_valid = error_term < 1

        new_tracks = []
        for track, (x, y), good_flag in zip(self.tracks, points_new.reshape(-1, 2), point_valid):
            # Track is good?
            if not good_flag:
                continue

            # New point is good, add to track.
            track.append((x, y))

            # Need to drop first old point?
            if len(track) > self.track_len:
                del track[0]

            # Track updated, add to track groups.
            new_tracks.append(track)

        # New track groups got, do update.
        self.tracks = new_tracks
        return points_old, points_new

    def get_new_tracks(self, frame, pt):
        """Get new tracks every detect_interval frames."""
        # Using mask to determine where to look for feature points.
        # mask = np.zeros_like(frame)
        # mask[roi[0]:roi[1], roi[2]:roi[3]] = 255

        # Get good feature points.
        feature_points = cv2.goodFeaturesToTrack(
            frame,25,0.01,5,useHarrisDetector=True, k=0.04)

        if feature_points is not None:
            for x, y in np.float32(feature_points).reshape(-1, 2):
                self.tracks.append([(x+pt[0], y+pt[1])])

    def get_average_track_length(self):
        """Get the average track length"""
        length = 0
        tracks = np.array(self.tracks)
        def distance(track):
            """Get distance between the first and last point."""
            delta_x = abs(track[-1][0] - track[0][0])
            delta_y = abs(track[-1][1] - track[0][1])
            return sqrt(delta_x*delta_x + delta_y*delta_y)
        for track in tracks:
            length += distance(track)
        return length / len(tracks)

    def draw_track(self, image):
        """Draw track lines on image."""
        cv2.polylines(image, [np.int32(track)
                              for track in self.tracks], False, (0, 255, 0))

def detect_one_face(im):
    frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]

def cornerDetection(cropImg, oriImg, pt):
	# grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cropImg = np.float32(cropImg)
	corners = cv2.goodFeaturesToTrack(cropImg, 25,0.01,5)
	corners = np.int0(corners)	#cast to int32 or int64

	#plot a dots on every corner detected
	for corner in corners:
		x, y = corner.ravel()
		cv2.circle(oriImg, (x+pt[0], y+pt[1]), 1, 255, -1)
	
	#return all the corner points
	return 
def applyGeometricTransformation(points_new, points_old, bbox):
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
    # for i in range(features.shape[0]):
        
    # non_zero_entries    = ~np.logical_and( (features[i,:,0]==0) ,(features[i,:,1] == 0) )
    features_p          = points_old[:,0,:]
    new_features_p      = points_new[:,0,:]
    tform               = transform.estimate_transform('similarity',features_p,new_features_p)  
    bbox = tform(bbox)

    return bbox

if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = frame_gray
    c, r, w, h  = detect_one_face(frame)
    bbox = np.array([(c,r), (c+w, r+h)])
    if w != 0 and h != 0:
        bbox_img = frame_gray[bbox[0,1] : bbox[1,1]+1 , bbox[0,0] : bbox[1,0]+1] 
    tracker = Tracker()
    kalman = Stabilizer(4, 2)
    global mp
    mp = np.array((2, 1), np.float32)  # measurement
    detect_interval = 5
    frame_idx = 0

    while(1):
        ret, frame = video.read()
        c, r, w, h  = detect_one_face(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # show the output frame
        if ret == False:
            break
        if w != 0 and h != 0:
            bbox = np.array([(c,r), (c+w, r+h)])
            bbox_img = frame_gray[bbox[0,1] : bbox[1,1]+1 , bbox[0,0] : bbox[1,0]+1]            
        #     template_corner = cornerDetection(bbox_img, frame, ([bbox[0,0],bbox[0,1]]))

            # Get new tracks every detect_interval frames.
            bbox_img = frame_gray[int(bbox[0,1]) : int(bbox[1,1])+1 , int(bbox[0,0]) : int(bbox[1,0])+1]  
            target_box = ([bbox[0,0],bbox[0,1]])
            if frame_idx % detect_interval == 0:
                tracker.get_new_tracks(bbox_img, target_box)


        # Update tracks.
        if len(tracker.tracks) > 0:
            points_old, points_new = tracker.update_tracks(prev_gray, frame_gray)

        bbox = applyGeometricTransformation(points_new, points_old, bbox)

        frame_idx += 1
        prev_gray = frame_gray
        # Draw tracks
        tracker.draw_track(frame)

        x1, y1 = tuple(bbox[0].astype(np.int32))
        x2, y2 = tuple(bbox[1].astype(np.int32))


        mp = np.array([[np.float32((x1+x2)/2)], [np.float32((y1+y2)/2)]])
        kalman.update(mp)
        point = kalman.prediction
        state = kalman.filter.statePost
        cv2.circle(frame, (int(state[0]), int(state[1])), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

        # display the bbox
        cv2.rectangle(frame, tuple(bbox[0].astype(np.int32)), tuple(bbox[1].astype(np.int32)), (255,0,0), thickness=2)
        cv2.rectangle(frame, (c, r), (c+w, r+h), (0, 255, 0), thickness=2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()