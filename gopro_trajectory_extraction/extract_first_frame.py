# extract the first frame of a mp4 file. used for detecting the position of camera_ori pose
import cv2
import sys

mp4_file_path = sys.argv[1]

cap = cv2.VideoCapture(mp4_file_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Read the first frame
ret, frame = cap.read()

# Check if the frame was successfully read
if not ret:
    print("Error: Unable to read the first frame.")
    cap.release()
    exit()

cap.release()

# Display or save the first frame
# For example, you can display the first frame using imshow
# cv2.imshow("gp_first_frame", frame)
# cv2.waitKey(0)  # Wait for any key to be pressed
# cv2.destroyAllWindows()
cv2.imwrite("gp_first_frame.png", frame)

