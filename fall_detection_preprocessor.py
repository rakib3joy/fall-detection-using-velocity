import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
import os
from pathlib import Path

# Keypoint indices from YOLOv8-pose
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
CONFIDENCE_THRESHOLD = 0.5

# YOLOv8-pose keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def compute_pixel_velocity(prev_y: float, curr_y: float, dt: float) -> float:
    pixel_displacement = curr_y - prev_y
    return pixel_displacement / dt

def choose_best_keypoint(keypoints_xy, keypoints_conf) -> list or None:
    if keypoints_conf is None or keypoints_xy is None:
        return None
    if len(keypoints_conf) <= RIGHT_SHOULDER or len(keypoints_xy) <= RIGHT_SHOULDER:
        return None
    
    left_valid = (
        keypoints_conf[LEFT_SHOULDER] > CONFIDENCE_THRESHOLD
        and not torch.isnan(keypoints_xy[LEFT_SHOULDER][1])
    )
    right_valid = (
        keypoints_conf[RIGHT_SHOULDER] > CONFIDENCE_THRESHOLD
        and not torch.isnan(keypoints_xy[RIGHT_SHOULDER][1])
    )
    
    if left_valid and right_valid:
        if keypoints_conf[LEFT_SHOULDER] >= keypoints_conf[RIGHT_SHOULDER]:
            return keypoints_xy[LEFT_SHOULDER].tolist()
        else:
            return keypoints_xy[RIGHT_SHOULDER].tolist()
    elif left_valid:
        return keypoints_xy[LEFT_SHOULDER].tolist()
    elif right_valid:
        return keypoints_xy[RIGHT_SHOULDER].tolist()
    
    return None

def extract_all_keypoints(frame, model) -> tuple:
    """Extract all 17 keypoints and return them along with the best shoulder keypoint"""
    result = model(frame, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None, None, None
    
    keypoints_xy = result.keypoints.xy[0]
    keypoints_conf = result.keypoints.conf[0] if result.keypoints.conf is not None else torch.ones(17)
    
    # Get the best shoulder keypoint for velocity calculation
    best_shoulder = choose_best_keypoint(keypoints_xy, keypoints_conf)
    
    # Only proceed if we have valid shoulder keypoints
    if best_shoulder is None:
        return None, None, None
    
    # Additional check: make sure we have at least some valid keypoints with decent confidence
    valid_keypoints = 0
    for i in range(min(17, len(keypoints_xy))):
        if (i < len(keypoints_xy) and i < len(keypoints_conf) and 
            not torch.isnan(keypoints_xy[i][0]) and not torch.isnan(keypoints_xy[i][1]) and
            keypoints_conf[i] > 0.3):  # At least 30% confidence
            valid_keypoints += 1
    
    # If we don't have at least 3 valid keypoints, skip this frame
    if valid_keypoints < 3:
        return None, None, None
    
    # Convert all keypoints to lists for CSV storage
    all_keypoints = []
    all_confidences = []
    
    # Ensure we always have exactly 17 keypoints
    for i in range(17):
        if i < len(keypoints_xy) and not torch.isnan(keypoints_xy[i][0]) and not torch.isnan(keypoints_xy[i][1]):
            all_keypoints.extend([keypoints_xy[i][0].item(), keypoints_xy[i][1].item()])
            all_confidences.append(keypoints_conf[i].item() if i < len(keypoints_conf) else 0.0)
        else:
            all_keypoints.extend([0.0, 0.0])  # Use 0,0 for missing keypoints
            all_confidences.append(0.0)
    
    return all_keypoints, all_confidences, best_shoulder

def process_video(video_path: str, model) -> tuple:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps
    
    results = []
    prev_shoulder_keypoint = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        all_keypoints, all_confidences, best_shoulder = extract_all_keypoints(frame, model)
        if all_keypoints is None:
            continue
        
        # Calculate velocity using shoulder keypoint
        velocity = 0.0
        if best_shoulder is not None and prev_shoulder_keypoint is not None:
            velocity = compute_pixel_velocity(prev_shoulder_keypoint[1], best_shoulder[1], dt)
            velocity = abs(velocity)
        
        # Create row with all keypoints, confidences, and velocity
        row = all_keypoints + all_confidences + [velocity]
        results.append(row)
        
        # Update previous shoulder keypoint for next iteration
        if best_shoulder is not None:
            prev_shoulder_keypoint = best_shoulder
    
    cap.release()
    return results, fps

def run_fall_detection(video_path: str, output_csv_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8x-pose.pt").to(device)
    
    print(f"Processing: {video_path}")
    print(f"Using device: {device.upper()}")
    
    data, fps = process_video(video_path, model)
    
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create column names for all keypoints (x, y coordinates)
    keypoint_columns = []
    for name in KEYPOINT_NAMES:
        keypoint_columns.extend([f"{name}_x", f"{name}_y"])
    
    # Create column names for all confidence scores
    confidence_columns = [f"{name}_conf" for name in KEYPOINT_NAMES]
    
    # Combine all column names
    all_columns = keypoint_columns + confidence_columns + ["vertical_velocity (pixel/s)"]
    
    # Check if any data was processed
    if len(data) == 0:
        print(f"Warning: No human detected in video {video_path}")
        print("Skipping CSV creation for this video")
        return
    
    # Debug information
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns expected: {len(all_columns)}")
    if len(data) > 0:
        print(f"Number of columns in first row: {len(data[0])}")
    
    df = pd.DataFrame(data, columns=all_columns)
    df.to_csv(output_csv_path, index=False)
    
    print(f"Done! Saved to: {output_csv_path}")
    print(f"FPS Detected: {fps:.2f}")
    print(f"Frames Processed: {len(data)}")

video_folder = r"./manual testing"

for video_file in Path(video_folder).glob("*.mp4"):
    video_path = str(video_file)
    print(f"Reading video: {video_path}")
    
    video_name = video_file.stem
    output_csv_path = f"testing_output2/{video_name}.csv"
    
    run_fall_detection(video_path, output_csv_path)