# get the reference T[marker_ref/camera] (camera's pose with marker as the origin)

import cv2
import numpy as np
import sys

# use the first frame of the realsense camera to detect the marker (assume realsense doesn't move)
# pic = "/mnt/robotics/marker_detect/calibration/sampled_img_realsense/1_rgb.png"
pic = sys.argv[1]


marker_size = 0.174 # [m]

# realsense at 640*480
camera_matrix = np.array([[600.41309815,   0.,         331.79178958], 
                          [  0.,         601.27047576, 252.95397012], 
                          [  0.,           0.,           1.        ]])
dist_coeffs = np.array([3.03606382e-03,  1.02851084e+00, -9.84561681e-04,  7.56707095e-03, -3.86339382e+00])

# gopro at video mode, use the first frame
# camera_matrix = np.array([[7.86035496e+02, 0.00000000e+00, 1.36152937e+03],
#                         [0.00000000e+00, 7.88728307e+02, 1.00635159e+03],
#                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# dist_coeffs = np.array([-0.34893022,  0.15111114,  0.00116502, -0.00176238, -0.0375606 ])


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters()
color_frame = cv2.imread(pic)

corners, ids, _ = cv2.aruco.detectMarkers(color_frame, dictionary, parameters=parameters)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(color_frame, corners, ids)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

    marker_rvec, marker_tvec = rvecs[0], tvecs[0]
    R, _ = cv2.Rodrigues(marker_rvec)  # [3, 3]
    t = marker_tvec.reshape((3, 1))
    
    ref_T = np.eye(4)
    ref_T[:3, :3] = R
    ref_T[:3, 3] = t.squeeze()
    ref_T = np.linalg.inv(ref_T)


    # GPT generated, untrustworthy
    # ref_t = -np.dot(R.T, t)
    # ref_T = np.eye(4)
    # ref_T[:3, :3] = R
    # ref_T[:3, 3] = ref_t.squeeze()

cv2.imshow("frame", color_frame)
print(ref_T)    # T[marker_ref/camera]
np.save("./T_matrix.npy", ref_T)

cv2.waitKey(0)