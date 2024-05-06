# get the reference T[marker_ref/camera] (camera's pose with marker as the origin)

import cv2
import numpy as np
import sys

# use the first frame of the realsense camera to detect the marker (assume realsense doesn't move)
# pic = "/mnt/robotics/marker_detect/calibration/sampled_img_realsense/1_rgb.png"
pic = sys.argv[1]

def get_reference_T_realsense(color_frame: np.ndarray, marker_size: float, camera_matrix=None, dist_coeffs=None, aruco_dict=cv2.aruco.DICT_7X7_50) -> np.ndarray:
    '''
    Calculate the T[custom_marker/realsense] at the given frame. \n
    color_frame: [W H C] realsense's frame. \n
    marker_size: marker's length. \n
    camera_matrix: [4, 4] realsense's intrinsic matrix. use default value if not given. \n
    dist_coeffs: [5] realsense's dist_coeffs. use default value if not given. \n
    aruco_dict: opencv's aruco dictionary. \n
    
    Return: T[custom_marker/realsense] or None
    '''

    # marker_size = 0.174 # [m]

    # realsense at 640*480
    if camera_matrix is None:
        camera_matrix = np.array([[600.41309815,   0.,         331.79178958], 
                          [  0.,         601.27047576, 252.95397012], 
                          [  0.,           0.,           1.        ]])

    if dist_coeffs is None:
        dist_coeffs = np.array([3.03606382e-03,  1.02851084e+00, -9.84561681e-04,  7.56707095e-03, -3.86339382e+00])


    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, _ = cv2.aruco.detectMarkers(color_frame, dictionary, parameters=parameters)

    if ids is not None:
        # cv2.aruco.drawDetectedMarkers(color_frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        marker_rvec, marker_tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(marker_rvec)  # [3, 3]
        t = marker_tvec.reshape((3, 1))
        
        ref_T = np.eye(4)
        ref_T[:3, :3] = R
        ref_T[:3, 3] = t.squeeze()
        ref_T = np.linalg.inv(ref_T)
        return ref_T
    
    else:
        print("No marker detected!")
        return
    

if __name__ == "__main__":
    pic = sys.argv[1]
    color_frame = cv2.imread(pic)
    ref_T = get_reference_T_realsense()(color_frame=color_frame, marker_size=0.174)
    print(ref_T)    # T[marker_ref/camera]
    np.save("T_matrix.npy", ref_T)
