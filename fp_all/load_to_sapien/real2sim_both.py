# show the trajectory of gripper and object in sapien using their tarjectory matrix T [N 4 4]

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import sys
from T_to_pq import T_to_pq
import pdb

def main(gp_T: np.ndarray, fp_T: np.ndarray):
    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.SapienRenderer()  # Create a renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine
    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency


    scene.add_ground(altitude=0)  # Add a ground
    actor_builder = scene.create_actor_builder()

    # actor_builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
    # actor_builder.add_box_visual(half_size=[0.05, 0.05, 0.05], color=[1., 0., 0.])
    # box = actor_builder.build_kinematic(name='box')  # Add a box (not affected by the gravity)
    # box.set_pose(sapien.Pose(p=[0, 0, 0.15]))

    actor_builder.add_collision_from_file("/mnt/data/cup_both/mesh/cup_resized.obj")
    
    actor_builder.add_visual_from_file("/mnt/data/cup_both/mesh/cup_resized.dae")
    cup = actor_builder.build_kinematic("cup")


    # loader = scene.create_urdf_loader()
    # loader.fix_root_link = False
    # robot = loader.load("./xarm_urdf/xarm7_gripper_only.urdf")
    # robot.set_name('xarm')

    actor_builder.add_collision_from_file("/mnt/data/umi_gripper_mesh/umi_gripper.obj")
    actor_builder.add_visual_from_file("/mnt/data/umi_gripper_mesh/umi_gripper.dae")
    robot = actor_builder.build_kinematic("robot")



    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene
    viewer.toggle_pause(True)   # pause on start
    

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    i = 0
    gp_T_total_frame_num = gp_T.shape[0]
    fp_T_total_frame_num = fp_T.shape[0]
    print(f"--- Gripper has {gp_T_total_frame_num} frames ---")
    print(f"--- Object has {fp_T_total_frame_num} frames ---")

    while not viewer.closed:  # Press key q to quit
        # scene.step()  # Simulate the world
        i += 1
        is_end = True

        if i < gp_T_total_frame_num:
            # pdb.set_trace()
            p, q = T_to_pq(gp_T[i])
            robot.set_pose(sapien.Pose(p=p, q=q))
            is_end = False
        
        if i < fp_T_total_frame_num:
            p, q = T_to_pq(fp_T[i])
            cup.set_pose(sapien.Pose(p=p, q=q))
            is_end = False
            
        if is_end:
            print("end")
            break
        
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':
    gp_T_path = sys.argv[1] # path to .npy file
    fp_T_path = sys.argv[2]

    gp_T = np.load(gp_T_path)
    fp_T = np.load(fp_T_path)

    main(gp_T, fp_T)