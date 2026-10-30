# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp #import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from thimble_v2_sim.assets.thimble_cfg import THIMBLE_V2_CFG

##
# Scene definition
##

FEET = ["foot_top","foot_top_1"]     # e.g. ".*_ankle_link" or ".*_foot"
BASE = "root"          # e.g. "base_link"
LEG_JOINTS = ["scapula_left", "hip_left", "roll_to_pitch_left", "knee_left", "ankle_left", "scapula_right", "hip_right", "roll_to_pitch_right", "knee_right", "ankle_right"]


@configclass
class ThimbleRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight = 1.0, #original, 0.25. better performance at 0.5
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 0.3,    # H1 uses 0.4 s; a small robot steps faster, originally 0.2 here
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func = mdp.track_lin_vel_xy_yaw_frame_exp,
        weight = 2.0,
        params={
            "std": 0.2,
            "command_name": "base_velocity",
        }
        )
    lin_vel_z_l2 = RewTerm(
        func = mdp.lin_vel_z_l2,
        weight = -1.0, # Penalty. Original was -2.0, adjusted to -0.5 (too small), adjusted to -1.0
        )
    flat_orientation_l2 = RewTerm(
        func = mdp.flat_orientation_l2,
        weight = -0.5,
        )
    # copy the remaining H1 terms here (feet_slide, joint limits, etc.),
    # replacing H1's body/joint names with yours


@configclass
class ThimbleRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ThimbleRewards = ThimbleRewards()

    def __post_init__(self):
        super().__post_init__()
        # rewards: base config's undesired_contacts targets ANYmal ".*THIGH" bodies
        # restricts policy to affect only the legs, NOT the head.
        self.actions.joint_pos.joint_names = LEG_JOINTS

        self.rewards.undesired_contacts = None

        # edit prebuilt rewards ; see IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py for a full list of inherited rewards.
        #self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        #self.rewards.track_lin_vel_xy_exp.params["std"] = 0.2 #0.2 is the STARTING value

        # robot
        self.scene.robot = THIMBLE_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + BASE

        # terminations: falling = base touches the ground
        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE]

        # events: the defaults are sized for 30+ kg robots
        self.events.add_base_mass = None
        self.events.base_com = None   # re-enable later, scaled to Thimble, bc normally for bigger robots
        self.events.push_robot = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE]

        # commands: small robot, small speeds
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class ThimbleFlatEnvCfg(ThimbleRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # observations: policy sees leg joints only
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        # observations: drop what the real robot can't measure
        self.observations.policy.base_lin_vel = None #Real thimble can't measure base linear velocity so REMOVE it


        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class ThimbleFlatEnvCfg_PLAY(ThimbleFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # observations: policy sees leg joints only
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False