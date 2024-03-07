import cv2 
from matplotlib import pyplot as plt
import os 
import sys

def capture_frame_and_save(folder_path, image_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera, exiting thread")
        sys.exit()
        return
    ret, frame = cap.read()
    cap.release()
    if ret:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, frame)
        print("Image saved successfully:", image_path)
    else:
        print("Error: Failed to capture frame")

capture_frame_and_save("./captures/", "1.jpg")
