# run all the post-process of a recorded demo in one file

import numpy as np
import sys
import cv2
import pdb
import os
import subprocess
from archive.utils import foundationpose_marker_as_origin, get_reference_T_realsense
from gripper_T_to_xarm_T import gripper_T_to_xarm_T
from get_gripper_openness import get_gripper_openness

def main(demo_path: str, ground_marker_size: float, mesh_folder_path: str, seq: list[str], cam_type: str):
    demo_path = os.path.abspath(demo_path)

    ################### mask generation ###################
    # press [r] to re-select object
    # press [esc] to confirm
    # os.chdir("/mnt/robotics/Blazar/vision/labeler")
    
    # cmd = ['python', 'label_seg.py', '--img_paths']
    # for object_i in seq:
    #     cmd.append(os.path.join(demo_path, object_i, "rgb/000000.png"))

    # subprocess.run(cmd, check=True)


    ################### run foundation pose ###################
    os.chdir("/mnt/robotics/FoundationPose-main")

    for object_i in seq:
        mesh_file_path = os.path.join(mesh_folder_path, object_i+".obj")
        cmd = ['bash', 'run_demo.sh', mesh_file_path, os.path.join(demo_path, object_i)]
        subprocess.run(cmd, check=True)


    ################### get transformation matrix of gripper and object ###################
    # pdb.set_trace()
    for object_i in seq:
        first_frame_path = os.path.join(demo_path, object_i, "rgb/000000.png")
        first_frame = cv2.imread(first_frame_path)

        ref_T = get_reference_T_realsense(first_frame, ground_marker_size, cam_type=cam_type)  # T[marker/object-first-frame]
        assert ref_T is not None

        fp_T = foundationpose_marker_as_origin(os.path.join(demo_path, object_i, "ob_in_cam"), ref_T)
        # fp_T = T_matrix_adjust(fp_T)
        np.save(os.path.join(demo_path, object_i+"_T.npy"), fp_T)


    ################### convert gripper pose from umi to xarm(soft) ###################
    xarmsoft_T = gripper_T_to_xarm_T(demo_path=demo_path)
    np.save(os.path.join(demo_path, "xarmsoft_T.npy"), xarmsoft_T)


    ################### calculate gripper openness ###################
    gripper_open_distance = get_gripper_openness(os.path.join(demo_path, "grip_cam"))   # gripper's openness in [cm]
    np.save(os.path.join(demo_path, "grip_openness.npy"), gripper_open_distance)    # [N]


if __name__ == "__main__":
    # demo_path = sys.argv[1]
    demo_path = "/mnt/robotics/fp_all/out/side-gripper-cam"
    mesh_folder_path = "/mnt/data/mesh"
    
    # objects that appear in the scene.
    # seq = ['gripper', 'cup']
    seq = ['gripper', 'box2']
    # seq = ['box2']

    # side camera used for recording the scene. not the gripper's camera (fixed to 435ca)
    # cam_type = "435i"
    # cam_type = "435ca"
    cam_type = "435"

    # main(demo_path=demo_path, ground_marker_size=0.174, mesh_folder_path=mesh_folder_path, seq=seq, cam_type=cam_type)

    for i in range(35, 50):
        demo_path_i = os.path.join(demo_path, str(i))
        print(i)
        main(demo_path=demo_path_i, ground_marker_size=0.174, mesh_folder_path=mesh_folder_path, seq=seq, cam_type=cam_type)


