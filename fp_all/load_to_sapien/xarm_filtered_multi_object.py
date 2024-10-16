# show the trajectory of gripper and objects in sapien using their tarjectory matrix T [N 4 4]
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import sys
import os
import pdb
# from gripper_pose_amend import umi_gripper_to_gripper_tip, xarm_gripper_tip_to_gripper
# from ..gripper_T_to_xarm_T import umi_gripper_to_gripper_tip

def T_to_pq(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # T: [4, 4]
    p = T[:3, 3]
    rotation_matrix = T[:3, :3]
    qw = np.sqrt(1 + rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2]) / 2
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
    q = np.array([qw, qx, qy, qz])
    return p, q


# def moving_average(data, window_size):
#     # data: [N 4 4] -> filtered_data: [Ni 4 4]
#     data = np.concatenate((R.from_matrix(data[:, :3, :3]).as_euler('xyz'), np.squeeze(data[:, :3, 3])), axis=1) # [N 6]
#     cumsum = np.cumsum(data, axis=0)
#     filtered_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

#     filtered_rot = R.from_euler('xyz', filtered_data[:, :3]).as_matrix()    # [Ni 3 3]
#     filtered_pos = filtered_data[:, 3:] # [Ni 3]
#     # pdb.set_trace()
#     filtered_data_matrix = np.concatenate([filtered_rot, filtered_pos[:, :, np.newaxis]], axis=2)  # [Ni 3 4]

#     pad = np.zeros([filtered_data_matrix.shape[0], 1, 4])  # [Ni 1 4]
#     pad[:, 0, -1] =1

#     filtered_data_matrix = np.concatenate([filtered_data_matrix, pad], axis=1)  # [Ni 4 4]
#     return filtered_data_matrix


def main(demo_path: str, mesh_folder_path: str, seq:list[str]):
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    scene.add_ground(altitude=0)  # Add a ground


    object_dict = {}
    object_T_dict = {}

    for object_i in seq:
        if object_i == "xarm":
            actor_builder = scene.create_actor_builder()
            loader = scene.create_urdf_loader()
            xarm = loader.load("/mnt/robotics/RobotLab/robotics/assets/mobile_sapien/xarm7/xarm_urdf/xarm7_gripper_only.urdf")
            xarm.set_name('xarm')
            object_dict[object_i] = xarm 

        elif object_i == "xarmsoft":
            _builder = scene.create_actor_builder()
            loader = scene.create_urdf_loader()
            xarm = loader.load("/mnt/robotics/ik/sapien_xarm7/xarm7_softfinger_gripper_only.urdf")
            xarm.set_name('xarm')
            object_dict[object_i] = xarm 
            
        else:
            object_i_mesh_path = os.path.join(mesh_folder_path, object_i+".obj")
            actor_builder = scene.create_actor_builder()
            actor_builder.add_visual_from_file(object_i_mesh_path)
            object_dict[object_i] = actor_builder.build_kinematic(object_i)
        
        object_i_T = np.load(os.path.join(demo_path, object_i+"_T.npy"))
        object_T_dict[object_i] = object_i_T

    
    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # viewer = Viewer(renderer)  # Create a viewer (window)
    viewer = Viewer()
    viewer.set_scene(scene)  # Bind the viewer and the scene
    viewer.paused = True
    

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-1, y=0, z=0.5)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(0.5, 1), y=0)

    # viewer.set_camera_xyz(x=0, y=-4, z=2)
    # viewer.set_camera_rpy(r=np.arctan2(2,4), p=0, y=0)

    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)


    for object_i, object_i_T in object_T_dict.items():
        object_i_frame_num = object_i_T.shape[0]
        print(f"--- {object_i} has {object_i_frame_num} frames ---")

        # filtered_T = moving_average(object_i_T, window_size=5)
        # object_T_dict[object_i] = filtered_T


    i = 0
    while not viewer.closed:  # Press key q to quit

        i += 1
        is_end = True

        for object_i, object_i_T in object_T_dict.items():
            # print(object_i)
            
            if i < object_i_T.shape[0]:
                p, q = T_to_pq(object_i_T[i])
                object_dict[object_i].set_pose(sapien.Pose(p=p, q=q))
                is_end = False


        if is_end:
            print("end")
            viewer.paused = True
            break
        
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':
    demo_path = sys.argv[1]
    mesh_folder_path = "/mnt/data/mesh"
    # seq = ['gripper', 'cup']
    # seq = ['gripper', 'cup', 'box']
    # seq = ['xarm']
    # seq = ['xarm', 'cup', 'box']
    seq = ['xarmsoft', 'box2']

    main(demo_path=demo_path, mesh_folder_path=mesh_folder_path, seq=seq)
