# show the trajectory of gripper and objects in sapien using their tarjectory matrix T [N 4 4]

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import sys
import os
from gripper_pose_amend import umi_gripper_to_gripper_tip, xarm_gripper_tip_to_gripper
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


def moving_average(data, window_size):
    # data: [N 16]
    cumsum = np.cumsum(data, axis=0)
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size



def main(eval_data: np.ndarray, eval_output: np.ndarray, mesh_folder_path: str, seq:list[str]):

    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.SapienRenderer()  # Create a renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine
    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency


    scene.add_ground(altitude=0)  # Add a ground


    object_dict = {}

    for object_i in seq:
        if object_i == "xarm":
            actor_builder = scene.create_actor_builder()
            loader = scene.create_urdf_loader()
            xarm = loader.load("/mnt/robotics/RobotLab/robotics/assets/mobile_sapien/xarm7/xarm_urdf/xarm7_gripper_only.urdf")
            xarm.set_name('xarm')
            object_dict[object_i] = xarm 
            
        else:
            object_i_mesh_path = os.path.join(mesh_folder_path, object_i+".obj")
            actor_builder = scene.create_actor_builder()
            actor_builder.add_visual_from_file(object_i_mesh_path)
            object_dict[object_i] = actor_builder.build_kinematic(object_i)

    
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_visual(half_size=np.array([0.01, 0.01, 0.01]), color=np.array([1, 0, 0]))
    tip = actor_builder.build_kinematic("tip")

    
    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene
    viewer.toggle_pause(True)   # pause on start
    

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-1, y=0, z=0.5)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(0.5, 1), y=0)

    # viewer.set_camera_xyz(x=0, y=-4, z=2)
    # viewer.set_camera_rpy(r=np.arctan2(2,4), p=0, y=0)

    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    
    if len(eval_data.shape) == 3:
        cup_pose = eval_data[0]
    else:
        cup_pose = eval_data
    
    p, q = T_to_pq(cup_pose)
    object_dict["cup"].set_pose(sapien.Pose(p=p, q=q))


    # pdb.set_trace()
    i = 0
    while not viewer.closed:  # Press key q to quit

        i += 1
        is_end = True


            
        if i < eval_output.shape[0]:
            p, q = T_to_pq(eval_output[i])
            object_dict["xarm"].set_pose(sapien.Pose(p=p, q=q))
            is_end = False

        if is_end:
            print("end")
            break
        
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':
    eval_data_path = sys.argv[1]
    eval_output_path = sys.argv[2]
    mesh_folder_path = "/mnt/data/mesh"
    seq = ['xarm', 'cup']

    eval_output_dict = np.load(eval_output_path, allow_pickle=True).item()
    eval_data_dict = np.load(eval_data_path, allow_pickle=True).item()

    # pdb.set_trace()
    scene_type = "single_initial"
    # scene_type = "single_along_with"
    # scene_type = "seq_initial"
    # scene_type = "seq_along_with"

    eval_data = eval_data_dict[scene_type]['cup']
    eval_output = eval_output_dict[scene_type]

    # pdb.set_trace()

    # debug only, diffusion can't always generate correct rotation
    gt_rotation = eval_data_dict[scene_type]['gripper']
    if len(gt_rotation.shape) == 3:
        gt_rotation = gt_rotation[0, :3, :3]
    else:
        gt_rotation = gt_rotation[:3, :3]

    eval_output[:, :3, :3] = gt_rotation


    main(eval_data=eval_data, eval_output=eval_output, mesh_folder_path=mesh_folder_path, seq=seq)
