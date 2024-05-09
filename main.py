import numpy as np
import sys
import cv2
import pdb

from foundation_pose_trajectory_extraction.foundationpose_marker_as_origin import foundationpose_marker_as_origin
from foundation_pose_trajectory_extraction.get_reference_T_realsense import get_reference_T_realsense
from gopro_trajectory_extraction.extract_first_frame import extract_first_frame
from gopro_trajectory_extraction.get_reference_T_gopro import get_reference_T_gopro
from gopro_trajectory_extraction.gopro_marker_as_origin import gopro_marker_as_origin


def main(gp_csv: str, gp_video: str, fp_path: str):
    gp_first_frame = extract_first_frame(gp_video)
    gp_ref_T = get_reference_T_gopro(gp_first_frame, 0.174)
    assert gp_ref_T is not None
    
    first_frame_idx = 140
    gp_T = gopro_marker_as_origin(gp_csv, gp_ref_T, first_frame_idx)
    np.save("gp_T.npy", gp_T)

    fp_first_frame = cv2.imread(fp_path+"/rgb/000000.png")
    fp_ref_T = get_reference_T_realsense(fp_first_frame, 0.174)
    assert fp_ref_T is not None

    fp_T = foundationpose_marker_as_origin(fp_path, fp_ref_T)
    np.save("fp_T.npy", fp_T)



if __name__ == "__main__":
    arg = sys.argv
    gp_csv = arg[1]
    gp_video = arg[2]
    fp_path = arg[3]
    main(gp_csv, gp_video, fp_path)