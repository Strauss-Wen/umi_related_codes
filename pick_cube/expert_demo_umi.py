from typing import Any, Dict, Union
import os

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("ExpertDemoUMIPick-v1" ,max_episode_steps=50)
class ExpertDemoUMIPickEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, save_single_traj="./robot_traj", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.init_qpos = np.array([0.0, 0.1963495, 0.0, -2.617993,
                                            0.0, 2.94155926, 0.78539816, 0.0, 0.0])
        self.robot_pose = [-0.16, -0.4, 0]
        self.cube_aim_position = [0, 0.3, 0.02]
        self.goal_radius = 0.08
        self.env = self

        self.traj_dest = save_single_traj
        if self.traj_dest:
            os.makedirs(self.traj_dest, exist_ok = True)
            print(f"File: {self.traj_dest} has been created")

        self.robot_pos = []
        self.robot_rot = []
        self.cube_pos = []
        self.cube_rot = []
        self.robot_grasp = []

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.robot_pos = []
        self.robot_rot = []
        self.cube_pos = []
        self.cube_rot = []
        self.robot_grasp = []
        
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
                    
        if self.traj_dest and is_obj_placed and is_robot_static:
            # save dictionary with poses at times to traj dest
            # look into saving as a .npy file
            with open(self.traj_dest + "/poses.npy", 'wb') as f:
                np.save(f, np.array(self.robot_pos))

            with open(self.traj_dest + "/rotations.npy", 'wb') as f:
                np.save(f, np.array(self.robot_rot))

            with open(self.traj_dest + "/grasp.npy", 'wb') as f:
                np.save(f, np.array(self.robot_grasp))

            with open(self.traj_dest + "/cube_pose.npy", 'wb') as f:
                np.save(f, np.array(self.cube_pos))

            with open(self.traj_dest + "/cube_rot.npy", 'wb') as f:
                np.save(f, np.array(self.cube_rot))

        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        if self.traj_dest:
            cur_pose = self.env.agent.tcp.pose.p[0]
            cur_rot = self.env.agent.tcp.pose.q[0]

            self.robot_pos.append(cur_pose.detach().cpu().numpy())
            self.robot_rot.append(cur_rot.detach().cpu().numpy())

            # save full cube trajectory, we will use initial position here as initial position when copying
            self.cube_pos.append(self.cube.pose.p[0].detach().cpu().numpy())
            self.cube_rot.append(self.cube.pose.q[0].detach().cpu().numpy())

            self.robot_grasp.append(info["is_grasped"].detach().cpu().numpy())

        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
