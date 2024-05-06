# extract the first frame of a mp4 file. used for detecting the position of camera_ori pose
import cv2
import sys
import numpy as np
# import pdb

def extract_first_frame(mp4_file_path: str) -> np.ndarray:
    """
    extract the first frame of a given mp4 file. \n
    Input: path to .mp4 file \n
    Return: numpy array typically with shape [W H C] \n
    """
    
    cap = cv2.VideoCapture(mp4_file_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        exit()

    cap.release()
    # pdb.set_trace()

    return frame

if __name__ == "__main__":
    mp4_file_path = sys.argv[1]
    first_frame = extract_first_frame(mp4_file_path=mp4_file_path)
    cv2.imwrite("gp_first_frame.png", first_frame)

