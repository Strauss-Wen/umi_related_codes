"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath("../xarm7")))

import numpy as np
import torch
import torch.random
import sapien
import sapien.render
from transforms3d.euler import euler2quat

from T_to_pq import *

from xarm7.xarm7 import XArm7
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq, XArm7Ability
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.building.ground import build_ground
    
@register_env("ExpertDemoFollow-v1", max_episode_steps=50)
class ExpertDemoEnv(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to push and move a cube to a goal region in front of it

    Randomizations
    --------------
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    Success Conditions
    ------------------
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.

    Visualization: https://maniskill.readthedocs.io/en/latest/tasks/index.html#pushcube-v1
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch", "xarm7_ability", "xarm7"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch, XArm7Ability]

    # set some commonly used values
    goal_thresh = 0.025
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="xarm7", robot_init_qpos_noise=0.02, traj='./robot_traj', rob_pos=None, max_reward=None, demo_loc=None, mesh_loc=None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.init_qpos = np.array([0.0, 0.1963495, 0.0, -2.617993,
                                0.0, 2.94155926, 0.78539816, 0.0, 0.0])
        self.robot_pose = rob_pos
        # self.robot_pose = [-0.16, -0.2, 0]
        self.env = self
        self.max_reward = max_reward # 6 with quat loss of some sort
        self.env_step = None
        self.last = None
        self.mesh_loc = mesh_loc

        # assert that needed data exists
        assert(os.path.exists(f"{demo_loc}"))
        assert(os.path.exists(f"{demo_loc}/cup_T.npy"))
        assert(os.path.exists(f"{demo_loc}/gripper_T.npy"))

        # load cube position, robot positions and rotation
        # cube_pose.npy  cube_rot.npy  poses.npy  rotations.npy
        self.cube_pose, self.cube_rot = T_to_pq(np.load(f'{demo_loc}/cup_T.npy'))
        self.rob_pose, self.rob_rot = T_to_pq(np.load(f'{demo_loc}/gripper_T.npy'))

        # optionally set the robot's initial position to match the gripper's
        # self.robot_pose = self.rob_pose[0]
        # self.robot_pose[-1] = 0

        super().__init__(*args, robot_uids=robot_uids, **kwargs) # robot_uids = robot_uids
        
    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    # fits pose trajectory arrays to correct dimension
    def fit_dim(self, poses):
        return torch.stack([torch.from_numpy(i).unsqueeze(dim=0).repeat((self.obj._num_objs, 1)) for i in poses]).to(self.device)

    def _load_scene(self, options: dict):
        # note: we dont need a table here
        # we just place our objects on the ground
        self.ground = build_ground(self.scene)

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        # self.obj = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        # )
        
        # make sure mesh files exist
        assert(os.path.exists(f"{self.mesh_loc}/cup_bottom_flattened.obj"))
        assert(os.path.exists(f"{self.mesh_loc}/cup_resized.dae"))
        
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(f"{self.mesh_loc}/cup_bottom_flattened.obj")
        builder.add_visual_from_file(f"{self.mesh_loc}/cup_resized.dae")
        self.obj = builder.build("cup")

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site) # will not show up

        # initialize our cube position tensor and put it on the gpu 
        with torch.device(self.device):
            # move trajectories to device and format them
            self.rob_pose = torch.from_numpy(self.rob_pose).to(self.device).float()
            self.rob_rot = torch.from_numpy(self.rob_rot).to(self.device).float()
            self.cube_pose = torch.from_numpy(self.cube_pose).to(self.device).float()
            self.cube_rot = torch.from_numpy(self.cube_rot).to(self.device).float()
            # self.rob_pose = torch.stack([torch.from_numpy(self.rob_pose)])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)

            # code to set position of the cube, we use half size to set z height
            # we run this on the ground so z=0 is the ground
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            
            # set initial object pose to match trajectory start
            obj_pose = Pose.create_from_pq(p=self.cube_pose[0], q=self.cube_rot[0]) 
            self.obj.set_pose(obj_pose)

            # create a marker for the goal position
            goal_xyz = self.cube_pose[-1]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            
            # finally set the qpos of the robot (can no longer do this due to mismatch in robots between demo and use)
            init_qpos = ([0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2] + [0] * 6)
            init_qpos = ([0.0, -0.4, 0.0, 0.5, 0.0, 0.9, -3.0] + [0] * 6)
            init_qpos = ([1.5, -0.4, 0.0, 0.5, 0.0, 0.9, -3.0] + [0] * 6)
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(init_qpos))
                )
                + init_qpos 
            )

            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose(self.robot_pose))

            # this may be a better pose to try
            # self.agent.robot.set_pose(sapien.Pose([-0.415, 0, 0])) 

            self.env_step = self.env.elapsed_steps.detach().clone()
            self.last = torch.zeros_like(self.env_step).to(self.device) - 1

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p - self.cube_pose[-1], axis=1
            )
            < self.goal_thresh
        )
        
        is_pose_same = torch.logical_and(
            torch.linalg.norm(self.agent.tcp.pose.p - self.rob_pose[-1], axis=1) < self.goal_thresh,
            self.env_step >= self.rob_pose.shape[0]
        )

        is_robot_static = self.agent.is_static(0.2)
        is_grasping = self.agent.is_grasping(self.obj)

        # "success": torch.logical_and(is_pose_same, ~(self.rob_grasp[-1].item() ^ self.agent.is_grasping(self.obj))),
        return {
            "success": is_obj_placed, # torch.logical_and(is_pose_same, is_obj_placed),
            "is_obj_placed": is_obj_placed,
            "is_pose_same": is_pose_same,
            "is_robot_static": is_robot_static,
            "is_grasping": is_grasping,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            is_grasping=info["is_grasping"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.cube_pose[-1].unsqueeze(0).repeat((info["is_grasping"].shape[0], 1)),
                obj_pose=self.obj.pose.raw_pose,
                step=self.elapsed_steps
            )
            # step=self.elapsed_steps[0].repeat((self.obj.pose.raw_pose.shape[0], 1)) # since we have conditioning
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # TODO: define reward function based on trajectory and timestep (info should have information about this)
        # TODO: take rotation into account with reward function as well as specific object size

        # reward for reaching the object
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # reward for keeping object still at final location
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward = static_reward * info["is_obj_placed"] 
        
        # sum of all rewards that dont depend on expert trajectory
        independent_reward = reaching_reward + static_reward*(self.max_reward - 1) # 5

        # grasping reward for grasping the object properly
        # independent_reward += info["is_grasping"] * torch.ones_like(independent_reward)

        # ignore trajectory reward for now
        # all_rewards = self.traj_reward(info).to(self.device)
        # mask_range = torch.arange(0, all_rewards.shape[1]).repeat((all_rewards.shape[0], 1)).to(self.device)
        # mask = self.env_step.unsqueeze(1) <= mask_range
        # max_step = torch.max(torch.where(mask, all_rewards, 0), axis=1)
        # self.env_step = max_step.indices + 1
        # reward = max_step.values + independent_reward

        reward = independent_reward

        # assign rewards to parallel environments that achieved success to the maximum of 4, as we now also consider the robot arm position reward
        reward[info["success"]] = self.max_reward
        return reward

    def traj_reward(self, info): # steps are the set of steps for each trajectory, 3
        # to measure distance between quaternions: first normalize each quaternion, then
        # angular diff = cos^-1 (2*<q1,q2>^2 - 1)
        # scaled between 0 and 1 difference: <q1,q2>^2 where 0 is for different quaternions and 1 is for similar
        # q_loss = torch.bmm(self.filter(steps, self.rob_rot).unsqueeze(1), self.obj.pose.q[...,:].unsqueeze(-1)).squeeze()
        # q_loss = self.quat_diff(self.cube_rot, self.obj.pose.q.unsqueeze(1))
        # reward = q_loss

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        obj_to_goal_dist = torch.linalg.norm(
                self.obj.pose.p.unsqueeze(1) - self.cube_pose, axis=2
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = place_reward #  * info['is_grasping'].unsqueeze(1) # obj pos change
        
        # finally assign a reward based on the robot arm position
        robot_to_path_dist = torch.linalg.norm(
                self.agent.tcp.pose.p.unsqueeze(1) - self.rob_pose, axis=2
        )
        alignment_reward = 1 - torch.tanh(5 * robot_to_path_dist)
        # reward += alignment_reward * info['is_grasping'].unsqueeze(1)

        return reward

    def quat_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Get the difference in radians between two quaternions.

        Args:
        a: first quaternion, shape (..., 4)
        b: second quaternion, shape (..., 4)
        Returns:
        Difference in radians, shape (N,)
        """

        # Normalize the quaternions
        a = a / torch.norm(a, dim=-1, keepdim=True)
        b = b / torch.norm(b, dim=-1, keepdim=True)

        # Compute the dot product between the quaternions
        dot_product = torch.sum(a * b, dim=-1)

        # Clamp the dot product to the range [-1, 1] to avoid numerical instability
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute the angle difference in radians
        # angle_diff = 2 * torch.acos(torch.abs(dot_product))

        return dot_product**2

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = self.max_reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
