# show the trajectory of gripper and objects in sapien using their tarjectory matrix T [N 4 4]

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import sys
import os

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


def gripper_to_gripper_tip(gripper_pose: np.ndarray) -> np.ndarray:
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
    # get the xarm's end effector's pose using the given xarm tip's pose
    
    x = 0
    y = 0
    z = -0.22

    # xyz_steps = [3, 0, 1]   # rotate along different axes 90 degrees at different steps

    # adj_x = np.array([[1, 0, 0, 0],
    #                   [0, 0, -1, 0],
    #                   [0, 1, 0, 0],
    #                   [0, 0, 0, 1]])

    # adj_y = np.array([[0, 0, 1, 0],
    #                   [0, 1, 0, 0],
    #                   [-1, 0, 0, 0],
    #                   [0, 0, 0, 1]])


    # adj_z = np.array([[0, -1, 0, 0],
    #                   [1, 0, 0, 0],
    #                   [0, 0, 1, 0],
    #                   [0, 0, 0, 1]])
    
    # rotate = np.eye(4)
    
    # for i in range(xyz_steps[0]):
    #     rotate = np.matmul(rotate, adj_x)

    # for i in range(xyz_steps[1]):
    #     rotate = np.matmul(rotate, adj_y)

    # for i in range(xyz_steps[2]):
    #     rotate = np.matmul(rotate, adj_z)

    # print(rotate)

    rotate = np.array([[ 0, -1,  0,  0,],
                         [ 0,  0,  1,  0,],
                         [-1,  0,  0,  0,],
                         [ 0,  0,  0,  1,]])


    adjust_T = np.array([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]])
    
    adjust_T = np.matmul(rotate, adjust_T)
    
    gripper_pose = np.matmul(gripper_tip_pose, adjust_T)
    return gripper_pose


def main(mesh_folder_path: str, seq:list[str]):

    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.SapienRenderer()  # Create a renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine
    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency


    scene.add_ground(altitude=0)  # Add a ground


    object_dict = {}
    object_T_dict = {}

    for object_i in seq:
        object_i_mesh_path = os.path.join(mesh_folder_path, object_i+".obj")
        actor_builder = scene.create_actor_builder()

        actor_builder.add_visual_from_file(object_i_mesh_path)

        object_dict[object_i] = actor_builder.build_kinematic(object_i)


    
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_visual(half_size=np.array([0.01, 0.01, 0.01]), color=np.array([1, 0, 0]))
    tip = actor_builder.build_kinematic("tip")

    actor_builder = scene.create_actor_builder()
    loader = scene.create_urdf_loader()
    xarm = loader.load("/mnt/data/sapien_xarm7/xarm7_softfinger_gripper_only.urdf")
    xarm.set_name('xarm')
    


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


    umi_pose = np.eye(4)
    
    xarm_pose = np.eye(4)
    # xarm_pose[2, -1] = 0.3

    while not viewer.closed:  # Press key q to quit

        p, q = T_to_pq(umi_pose)
        object_dict['gripper'].set_pose(sapien.Pose(p=p, q=q))
    
        
        tip_pose = gripper_to_gripper_tip(umi_pose)
        p_tip, q_tip = T_to_pq(tip_pose)
        tip.set_pose(sapien.Pose(p=p_tip, q=q_tip))

        xarm_pose = xarm_gripper_tip_to_gripper(tip_pose)
        p_xarm, q_xarm = T_to_pq(xarm_pose)
        xarm.set_pose(sapien.Pose(p=p_xarm, q=q_xarm))

        
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':
    # demo_path = sys.argv[1]
    mesh_folder_path = "/mnt/data/mesh"
    seq = ['gripper']
    # seq = ['gripper', 'cup', 'box']

    main(mesh_folder_path=mesh_folder_path, seq=seq)