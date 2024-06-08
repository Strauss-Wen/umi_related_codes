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

import numpy as np
import torch
import torch.random
import sapien
import sapien.render
from transforms3d.euler import euler2quat

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

@register_env("ExpertDemo-v2", max_episode_steps=50)
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
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="xarm7", robot_init_qpos_noise=0.02, traj='./robot_traj', **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.init_qpos = np.array([0.0, 0.1963495, 0.0, -2.617993,
                                0.0, 2.94155926, 0.78539816, 0.0, 0.0])
        self.robot_pose = [-0.16, -0.4, 0]
        self.cube_aim_position = [0, 0.3, 0.02]
        self.goal_radius = 0.08
        self.env = self
        self.max_reward = 3
        self.env_step = None
        self.last = None

        if traj and os.path.exists(traj):
            # load cube position, robot positions and rotation
            # cube_pose.npy  cube_rot.npy  poses.npy  rotations.npy
            self.cube_pose = np.load(traj+'/cube_pose.npy')
            self.cube_rot = np.load(traj+'/cube_rot.npy')
            self.rob_pose = np.load(traj+'/poses.npy')
            self.rob_rot = np.load(traj+'/rotations.npy')

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
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # initialize our cube position tensor and put it on the gpu 
        with torch.device(self.device):
            self.cube_aim_position = torch.unsqueeze(torch.tensor(self.cube_aim_position), dim=0)
            self.cube_aim_position = self.cube_aim_position.repeat((self.obj._num_objs, 1))

            # move trajectories to device and format them
            self.rob_pose = torch.from_numpy(self.rob_pose).to(self.device)
            self.rob_rot = torch.from_numpy(self.rob_rot).to(self.device)
            self.cube_pose = torch.from_numpy(self.cube_pose).to(self.device)
            self.cube_rot = torch.from_numpy(self.cube_rot).to(self.device)
            # self.rob_pose = torch.stack([torch.from_numpy(self.rob_pose)])
            '''
            self.rob_pose = self.fit_dim(self.rob_pose)
            self.rob_rot = self.fit_dim(self.rob_rot)
            self.cube_pose = self.fit_dim(self.cube_pose)
            self.cube_rot = self.fit_dim(self.cube_rot)
            '''

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)

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
            
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=self.cube_pose[0], q=self.cube_rot[0]) # set initial cube pose to match trajectory start
            self.obj.set_pose(obj_pose)

            # finally set the qpos of the robot (can no longer do this due to mismatch in robots between demo and use)
            '''
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(self.init_qpos))
                )
                + self.init_qpos 
            )

            self.env.agent.reset(qpos)
            '''
            self.env.agent.robot.set_pose(sapien.Pose(self.robot_pose))

            self.env_step = self.env.elapsed_steps.detach().clone()
            self.last = torch.zeros_like(self.env_step).to(self.device) - 1

    def evaluate(self):
        # TODO: redefine success based on whether final pose of cube matches desired target position and ROTATION
        # use self.env.elapsed_steps to get the index of correct robot and cube position
        # can define success as if elapsed steps equivalent to length of the cube array + cube position is similar
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.cube_pose[-1,:2], axis=1
            )
            < self.goal_radius
        )

        
        is_pose_same = torch.logical_and(
            torch.linalg.norm(self.agent.tcp.pose.p - self.filter(self.last, self.rob_pose), axis=1) < self.goal_radius,
            self.env_step >= self.rob_pose.shape[0]
        )

        return {
            "success": is_pose_same,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.cube_aim_position,
                obj_pose=self.obj.pose.raw_pose,
                step=self.elapsed_steps
            )
            # step=self.elapsed_steps[0].repeat((self.obj.pose.raw_pose.shape[0], 1)) # since we have conditioning
        return obs

    # select the correct value in the traj
    def filter(self, env_steps, traj):
        to_stack = [traj[i if i < traj.shape[0] else -1] for i in env_steps]

        return torch.stack(to_stack)


    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # TODO: define reward function based on trajectory and timestep (info should have information about this)
        # TODO: take rotation into account with reward function as well as specific object size

        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        '''
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # see if cube is at final position
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.cube_pose[-1,:,:2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached # initially was +=, now ignoring the reaching reward
        '''

        # return now if we cant copy demo anymore
        if self.env.elapsed_steps[0] >= self.cube_pose.shape[0]: # or self.env.elapsed_steps[0] < 8
            cur_step = -1
        else:
            cur_step = self.env.elapsed_steps[0]

        all_rewards = self.traj_reward(tcp_to_push_pose_dist).to(self.device)
        mask_range = torch.arange(0, all_rewards.shape[1]).repeat((all_rewards.shape[0], 1)).to(self.device)
        mask = self.env_step.unsqueeze(1) < mask_range
        max_step = torch.max(torch.where(mask, all_rewards, -1 * (self.max_reward + 1)), axis=1)
        self.env_step = max_step.indices
        reward = max_step.values

        # assign rewards to parallel environments that achieved success to the maximum of 4, as we now also consider the robot arm position reward
        reward[info["success"]] = self.max_reward
        return reward

    def traj_reward(self, tcp_to_push_pose_dist): # steps are the set of steps for each trajectory
        # to measure distance between quaternions: first normalize each quaternion, then
        # angular diff = cos^-1 (2*<q1,q2>^2 - 1)
        # scaled between 0 and 1 difference: <q1,q2>^2 where 0 is for different quaternions and 1 is for similar
        # q_loss = torch.bmm(self.filter(steps, self.rob_rot).unsqueeze(1), self.obj.pose.q[...,:].unsqueeze(-1)).squeeze()
        q_loss = self.quat_diff(self.rob_rot, self.obj.pose.q.unsqueeze(1))
        reward = q_loss

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        # TODO: maybe give n steps before agent has to start copying the teacher
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
                self.obj.pose.p.unsqueeze(1)[..., :2] - self.cube_pose[..., :2], axis=2
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached.unsqueeze(1)
        
        # finally assign a reward based on the robot arm position
        robot_to_path_dist = torch.linalg.norm(
                self.agent.tcp.pose.p.unsqueeze(1) - self.rob_pose, axis=2
        )
        reaching_reward = 1 - torch.tanh(5 * robot_to_path_dist)
        reward += reaching_reward

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
