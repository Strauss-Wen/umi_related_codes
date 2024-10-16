import numpy as np
import os
import sys

def umi_gripper_to_gripper_tip(gripper_pose: np.ndarray) -> np.ndarray:
    # get the gripper tip's T matrix at this step using the given gripper's T
    
    # unit: [m]
    x = 0.045
    y = 0.18
    z = 0

    adjust_T = np.array([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]])
    
    gripper_tip_pose = np.matmul(gripper_pose, adjust_T)
    return gripper_tip_pose

def xarm_gripper_tip_to_gripper(gripper_tip_pose: np.ndarray) -> np.ndarray:
    # get the xarm(soft)'s end effector's pose using the given xarm tip's pose
    rotate_T = np.array([[ 0, -1,  0,  0,],
                         [ 0,  0,  1,  0,],
                         [-1,  0,  0,  0,],
                         [ 0,  0,  0,  1,]])
    
    x = 0
    y = 0
    z = -0.22

    adjust_T = np.array([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]])
    
    adjust_T = np.matmul(rotate_T, adjust_T)
    gripper_T = np.matmul(gripper_tip_pose, adjust_T)

    return gripper_T

def gripper_T_to_xarm_T(demo_path: str) -> np.ndarray:
    gripper_T = np.load(os.path.join(demo_path, "gripper_T.npy"))
    xarm_T = []

    frame_num = gripper_T.shape[0]
    for i in range(frame_num):
        umi_T_i = gripper_T[i]
        gripper_tip_i = umi_gripper_to_gripper_tip(gripper_pose=umi_T_i)
        xarm_T_i = xarm_gripper_tip_to_gripper(gripper_tip_pose=gripper_tip_i)
        xarm_T.append(xarm_T_i)

    xarm_T = np.array(xarm_T)
    
    return xarm_T

if __name__ == "__main__":
    # process single demo
    # demo_path = sys.argv[1]
    # gripper_T_to_xarm_T(demo_path=demo_path)

    # process demos
    demos_parent_path = sys.argv[1]
    demo_names = os.listdir(demos_parent_path)
    for i in range(len(demo_names)):
        demo_path = os.path.join(demos_parent_path, demo_names[i])
        xarm_T = gripper_T_to_xarm_T(demo_path=demo_path)
        np.save(os.path.join(demo_path, "xarmsoft_T.npy"), xarm_T)
        print(demo_path)
        


