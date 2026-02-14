
#Goal:
#Speed = displacement / time.

# Speed (m/s) = Displacement (pixels/frame)×FPS (frames/sec)​ / pixel per frame.

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
from collections import deque
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import csv

# Load YOLOv8 Pose Model
yolo_model = YOLO('yolov8x-pose.pt')

# Constants
camera_fps = 30

# Moving Average Filter
displacement_history = deque(maxlen=5)

def smooth_displacement(value):
    displacement_history.append(value)
    return np.mean(displacement_history)

# YOLO Keypoints Indexes (COCO format)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

# Kalman Filter Initialization
def initialize_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])
    kf.P *= 1000
    kf.R = np.eye(2) * 10
    kf.Q = np.eye(4) * 0.5
    return kf

# Dynamic ppm Estimation
def get_pixels_per_meter(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return None

    nose = keypoints[NOSE][:2]
    l_ankle = keypoints[L_ANKLE][:2]
    r_ankle = keypoints[R_ANKLE][:2]
    ankle = (l_ankle + r_ankle) / 2

    pixel_height = math.dist(nose, ankle)
    if pixel_height == 0:
        return None

    # Assume real human height ~1.70m
    real_world_height = 1.70
    real_shoulder_width = real_world_height * 0.24

    l_shoulder = keypoints[L_SHOULDER][:2]
    r_shoulder = keypoints[R_SHOULDER][:2]
    pixel_shoulder_width = math.dist(l_shoulder, r_shoulder) # measure shoulder width
    if pixel_shoulder_width == 0:
        return None
    
#pixel per meter
    ppm_from_height = pixel_height / real_world_height
    ppm_from_shoulders = pixel_shoulder_width / real_shoulder_width

    ppm = (ppm_from_height + ppm_from_shoulders) / 2

    if ppm < 50 or ppm > 1000:
        return None

    return ppm

# Calculate displacement
def get_combined_displacement(keypoints, prev_positions):# compute movement(displacement) # that means how much far the person from prev frame to current frame
    if keypoints is None or len(keypoints) < 17:# if missing th keypoints
        return 0, 0

    key_indices = [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]# select body part
    current_positions = np.mean([keypoints[i][:2] for i in key_indices], axis=0) # gives on point average

    if prev_positions is None:
        return 0, 0

    displacement = np.abs(current_positions - prev_positions)
    return displacement[1], displacement[0]#vetical and horizontal

# Draw keypoints
def draw_keypoints(frame, keypoints):
    for point in keypoints:
        if len(point) >= 2:
            x, y = map(int, point[:2])
            conf = point[2] if len(point) > 2 else 1.0
            if conf > 0.6:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame

# Folder setup
video_folder = './New folder (3)'
output_csv_folder = './csv_testsssss'
os.makedirs(output_csv_folder, exist_ok=True)

video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path) # read frames from the video
    frame_count = 0 # which frame we're on. #used to track time (time = frame_count / FPS)

    kf = initialize_kalman() # Kalman filter for the current video
    prev_positions = None
    csv_data = [] #frame, time, speed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame) #holds outputs for each detected person# all people
        keypoints = None 
        for r in results:
            if r.keypoints and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy[0].cpu().numpy()
                if keypoints.shape[0] >= 17:
                    break

        if keypoints is None:# when can not detect a person
            frame_count += 1
            continue

        frame = draw_keypoints(frame, keypoints)

        vert_disp, horiz_disp = get_combined_displacement(keypoints, prev_positions) # compare keypoints current frame vs prev frame
        # how much vrtical and horizontal displace I get.

        z = np.array([[horiz_disp], [vert_disp]])# just observed
        kf.predict()# predict displacement when keypoints missing # predict based on precious displacement
        kf.update(z)# correct that guess  one using

        corrected_disp_x = smooth_displacement(abs(kf.x[0]))# estimated displacement of x
        corrected_disp_y = smooth_displacement(abs(kf.x[2]))# estimated displacement of y


        # Speed calculation
        ppm = get_pixels_per_meter(keypoints)
        if ppm is not None and ppm != 0:
            speed_mps = corrected_disp_x * camera_fps / ppm
        else:
            speed_mps = 0


        timestamp = frame_count / camera_fps
        csv_data.append([frame_count, round(timestamp, 2), round(float(speed_mps), 4)])

        prev_positions = np.mean([keypoints[i][:2] for i in [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]], axis=0)

        cv2.putText(frame, f'Speed: {round(float(speed_mps), 2)} m/s', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Optional display
        # cv2.imshow('YOLOv8 Displacement Tracker', frame)
        frame_count += 1

        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break

    cap.release()

    csv_filename = os.path.splitext(video_file)[0] + '_velocity.csv'
    csv_path = os.path.join(output_csv_folder, csv_filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Time (s)', 'Speed (m/s)'])
        writer.writerows(csv_data)

    print(f"Saved velocity data to {csv_path}")

cv2.destroyAllWindows()





