import copy
import os.path as osp
from typing import Dict, List, Union

import numpy as np
import sapien
from sympy import Tuple
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (PDJointPosControllerConfig,
                                           PDJointPosMimicController,
                                           PDJointPosMimicControllerConfig)
from mani_skill.agents.controllers.base_controller import (ControllerConfig,
                                                           DictController)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors import BaseSensorConfig, CameraConfig
import sapien.physx as physx
from mani_skill.utils.structs import Actor
from mani_skill.utils import sapien_utils, common
from transforms3d.euler import euler2quat

@register_agent()
class XArm7(BaseAgent):
    uid = "xarm7"
    # TODO model the urdf config right
    urdf_config = dict(
                    _materials=dict(
                        gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
                    ),
                    link=dict(
                        left_finger=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
                        right_finger=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05)
                    )
                )
    urdf_path = osp.join(
        osp.dirname(__file__), "sapien_xarm7/xarm_urdf/xarm7_gripper.urdf"
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -0.4,
                    0.0,
                    0.5,
                    0.0,
                    0.9,
                    -3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    '''
    @property
    def _sensor_configs(self) -> List[BaseSensorConfig]:
        # TODO if there are mounted cameras on the robot add them and mount them here
        return [
            CameraConfig("base_camera", pose=sapien.Pose(p=[-0.06, 0, 0.03], q=euler2quat(0, -np.pi/2, 0)), width=128, height=128, fov=np.pi / 2, near=0.01, far=100, entity_uid="link_eef")
        ]
    '''

    @property
    def _controller_configs(self) -> Dict[str, Union[ControllerConfig,]]:
        arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        arm_stiffness = 2e4 # 2e4
        arm_damping = 1e3 # 1e3
        arm_force_limit = 500 # 200

        gripper_joint_names = [
            "left_inner_knuckle_joint",
            "left_outer_knuckle_joint",
            "left_finger_joint",
            "right_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint"
        ]
        gripper_stiffness = 1e3 # 5e3
        gripper_damping = 1e2 # 10
        gripper_force_limit = 100 # 1e5

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            arm_joint_names,
            lower=[-3.14],
            upper=[3.14],
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            arm_joint_names,
            -0.1,
            0.1,
            arm_stiffness,
            arm_damping,
            arm_force_limit,
            use_delta=True,
            drive_mode="acceleration",
        )
        arm_pd_joint_target_delta_pos = copy.deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # Gripper
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            gripper_joint_names,
            -0.01, # have force when object is thin
            0.85, # 0.85
            gripper_stiffness,
            gripper_damping,
            gripper_force_limit,
            drive_mode="acceleration"
        )

        return dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos, balance_passive_force=True
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos, balance_passive_force=True),
        )

    def _after_loading_articulation(self):
        robot = self.robot
        self.lf = robot.find_link_by_name("left_finger")
        self.rf = robot.find_link_by_name("right_finger")

        lik = robot.find_link_by_name("left_inner_knuckle")
        lok = robot.find_link_by_name("left_outer_knuckle")
        rik = robot.find_link_by_name("right_inner_knuckle")
        rok = robot.find_link_by_name("right_outer_knuckle")
        left_p_f = [-2.9802322e-08, 3.5464998e-02, 4.2039029e-02]
        left_p_p = [0.0, -0.01499999, 0.01500002]
        right_p_f = [2.9802322e-08, -3.5464998e-02, 4.2039003e-02]
        right_p_p = [1.4901161e-08, 1.5000006e-02, 1.4999989e-02]
        drive = self.scene.create_drive(
            lik, sapien.Pose(left_p_f), self.lf, sapien.Pose(left_p_p)
        )
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        drive = self.scene.create_drive(
            rik, sapien.Pose(right_p_f), self.rf, sapien.Pose(right_p_p)
        )
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        link_eef = robot.find_link_by_name("link_eef")
        link7 = robot.find_link_by_name("link7")
        ee_base_link = robot.find_link_by_name("xarm_gripper_base_link")
        
        '''
        self.lf = robot.find_link_by_name("left_outer_knuckle")
        self.rf = robot.find_link_by_name("right_outer_knuckle")
        '''

        # NOTE: Currently ManiSkill does not have a simple way to manage collision groups in a batched manner.
        for link in [lik, lok, self.lf, rik, rok, self.rf, link_eef, link7, ee_base_link]:
            if link is not None:
                for obj in link._objs:
                    for s in obj.collision_shapes:
                        s.set_collision_groups([1, 1, 1 << 2, 0])

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()
        self.tcp = robot.find_link_by_name("link_tcp")

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6] # last 6 joints are for the gripper
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    '''
    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.lf, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.rf, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.lf.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.rf.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
    '''
    
    # TODO/NOTE: Checking contact pair forces/impulses is unfortunately still rather verbose, and is quite different on GPU vs CPU.
    # this code is copied directly from panda implementation in ManiSkill
    def is_grasping(self, object: Actor = None, min_impulse=1e-6, max_angle=85): # max_angle was 85
        if physx.is_gpu_enabled():
            if object.name not in self.queries:
                body_pairs = list(zip(self.lf._bodies, object._bodies))
                body_pairs += list(zip(self.rf._bodies, object._bodies))
                self.queries[object.name] = (
                    self.scene.px.gpu_create_contact_pair_impulse_query(body_pairs),
                    (len(object._bodies), 3),
                )
            query, contacts_shape = self.queries[object.name]
            self.scene.px.gpu_query_contact_pair_impulses(query)
            # query.cuda_contacts # (num_unique_pairs * num_envs, 3)
            contacts = (
                query.cuda_impulses.torch().clone().reshape((-1, *contacts_shape))
            )
            lforce = torch.linalg.norm(contacts[0], axis=1)
            rforce = torch.linalg.norm(contacts[1], axis=1)

            # NOTE (stao): 0.5 * time_step is a decent value when tested on a pick cube task.
            min_force = 0.5 * self.scene.px.timestep

            # direction to open the gripper
            ldirection = self.lf.pose.to_transformation_matrix()[..., :3, 1]
            rdirection = -self.rf.pose.to_transformation_matrix()[..., :3, 1]
            langle = common.compute_angle_between(ldirection, contacts[0])
            rangle = common.compute_angle_between(rdirection, contacts[1])
            lflag = torch.logical_and(
                lforce >= min_force, torch.rad2deg(langle) <= max_angle
            )
            rflag = torch.logical_and(
                rforce >= min_force, torch.rad2deg(rangle) <= max_angle
            )

            return torch.logical_and(lflag, rflag)
        else:
            contacts = self.scene.get_contacts()

            if object is None:
                finger1_contacts = sapien_utils.get_actor_contacts(
                    contacts, self.lf._bodies[0].entity
                )
                finger2_contacts = sapien_utils.get_actor_contacts(
                    contacts, self.rf._bodies[0].entity
                )
                return (
                    np.linalg.norm(sapien_utils.compute_total_impulse(finger1_contacts))
                    >= min_impulse
                    and np.linalg.norm(
                        sapien_utils.compute_total_impulse(finger2_contacts)
                    )
                    >= min_impulse
                )
            else:
                limpulse = sapien_utils.get_pairwise_contact_impulse(
                    contacts,
                    self.lf._bodies[0].entity,
                    object._bodies[0].entity,
                )
                rimpulse = sapien_utils.get_pairwise_contact_impulse(
                    contacts,
                    self.rf._bodies[0].entity,
                    object._bodies[0].entity,
                )

                # direction to open the gripper
                ldirection = self.lf.pose.to_transformation_matrix()[
                    ..., :3, 1
                ]
                rdirection = -self.rf.pose.to_transformation_matrix()[
                    ..., :3, 1
                ]

                # angle between impulse and open direction
                langle = common.np_compute_angle_between(ldirection[0], limpulse)
                rangle = common.np_compute_angle_between(rdirection[0], rimpulse)

                lflag = (
                    np.linalg.norm(limpulse) >= min_impulse
                    and np.rad2deg(langle) <= max_angle
                )
                rflag = (
                    np.linalg.norm(rimpulse) >= min_impulse
                    and np.rad2deg(rangle) <= max_angle
                )

                return torch.tensor([all([lflag, rflag])], dtype=bool)
