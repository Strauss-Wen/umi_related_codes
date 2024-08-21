# show the trajectory of gripper and objects in sapien using their tarjectory matrix T [N 4 4]

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import sys
import os
import pdb


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


def main(eval_output: dict, eval_scene_path: str, scene_type: str, mesh_folder_path: str, seq:list[str]):
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    scene.add_ground(altitude=0)  # Add a ground

    object_dict = {}

    for object_i in seq:
        if object_i == "xarm":
            actor_builder = scene.create_actor_builder()
            loader = scene.create_urdf_loader()
            xarm = loader.load("/mnt/robotics/ik/sapien_xarm7/xarm7_softfinger_gripper_only.urdf")
            xarm.set_name('xarm')
            object_dict[object_i] = xarm
            
        else:
            object_i_mesh_path = os.path.join(mesh_folder_path, object_i+".obj")
            actor_builder = scene.create_actor_builder()
            actor_builder.add_visual_from_file(object_i_mesh_path)
            object_dict[object_i] = actor_builder.build_kinematic(object_i)

    box2_traj = np.load(os.path.join(eval_scene_path, "box2_T.npy"))

    
    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer()  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene
    viewer.paused = True
    
    viewer.set_camera_xyz(x=-1, y=0, z=0.5)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(0.5, 1), y=0)

    # viewer.set_camera_xyz(x=0, y=-4, z=2)
    # viewer.set_camera_rpy(r=np.arctan2(2,4), p=0, y=0)

    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    # pdb.set_trace()
    eval_grip_traj = eval_output[scene_type]['gripper_traj']
    eval_grip_open = eval_output[scene_type]['openness']


    # pdb.set_trace()
    i = 0
    while not viewer.closed:  # Press key q to quit

        i += 1
        is_end = True


        if i < eval_grip_traj.shape[0]:
            p, q = T_to_pq(eval_grip_traj[i])
            object_dict["xarm"].set_pose(sapien.Pose(p=p, q=q))
            
            p, q = T_to_pq(box2_traj[i-1])
            object_dict["box2"].set_pose(sapien.Pose(p=p, q=q))
            
            is_end = False

        if is_end:
            print("end")
            break
    
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':
    mesh_folder_path = "/mnt/data/mesh"
    seq = ['xarm', 'box2']


    # scene_type = "single_initial"
    # scene_type = "single_along_with"
    scene_type = "seq_initial"
    # scene_type = "seq_along_with"


    eval_output = np.load("/mnt/robotics/ik/eval_output/eval_eval_data_with_obj_traj.npy", allow_pickle=True).item()
    eval_scene_path = "/mnt/robotics/ik/out/side-gripper-cam/0"
    # eval_data = eval_data[0]
    # pdb.set_trace()


    main(eval_output=eval_output, eval_scene_path=eval_scene_path, scene_type=scene_type, mesh_folder_path=mesh_folder_path, seq=seq)
