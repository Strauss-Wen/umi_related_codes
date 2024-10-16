import numpy as np
import cv2
import os
import pdb

def get_gripper_openness(img_folder_path: str) -> np.ndarray:
    # use D435ca's default config

    fx = 606.62530518
    fy = 606.92730713
    cx = 321.48971558
    cy = 248.01927185
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    k3 = 0

    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    marker_size = 1.6 # [cm]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    img_names = os.listdir(img_folder_path)
    img_names.sort()
    distance = []

    for img_name in img_names:
        img = cv2.imread(os.path.join(img_folder_path, img_name))

        corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

        try:
            id_0 = int(np.where(ids[:, 0] == 0)[0])
            id_1 = int(np.where(ids[:, 0] == 1)[0])

            corners_0_1 = (corners[id_0], corners[id_1])
            
        except:
            distance.append(distance[-1])
            continue


        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners_0_1, marker_size, camera_matrix, dist_coeffs)
        # tvecs: [N 1 3]

        t_diff = tvecs[1, 0, 0] - tvecs[0, 0, 0]
        distance_i = (abs(t_diff) - 4.7)
        distance_i = distance_i if distance_i > 0 else 0

        distance.append(distance_i)
    
    distance = np.array(distance)

    return distance


if __name__ == "__main__":
    img_folder_path = "./sampled_photos"
    distance = get_gripper_openness(img_folder_path=img_folder_path)
    print(distance)
