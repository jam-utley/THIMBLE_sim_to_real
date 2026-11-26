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

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg #import terminations for values that explode unnescesarily
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
    termination_penalty = RewTerm(
        func=mdp.is_terminated, 
        weight=-200.0,
        )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight = 3.0, #original, 0.25. better performance at 0.5. Increased from 2->10
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 0.25,    # H1 uses 0.4 s; a small robot steps faster, originally 0.2 here
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func = mdp.track_lin_vel_xy_yaw_frame_exp,
        weight = 1.5, #originally 2.0
        params={
            "std": 0.2,
            "command_name": "base_velocity",
        }
        )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,  
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(
        func = mdp.lin_vel_z_l2,
        weight = -2.0, # Penalty. Original was -2.0, adjusted to -0.5 (too small), adjusted to -1.0
        )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) 
    flat_orientation_l2 = RewTerm(
        func = mdp.flat_orientation_l2,
        weight = -0.6,
        )
    #to keep scapula/hips roughly neutral. 
    scapula_retain_orientation = RewTerm(
        func = mdp.joint_deviation_l1, # joint_deviation_l1 originally.
        weight = -0.2,
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=["scapula_left", "scapula_right"])
        }
        )
    #Penalty for foot sliding
    feet_slide = RewTerm(
        func = mdp.feet_slide,
        weight = -0.4,
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names = FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names = FEET)
        }
        )
    foot_clearance = RewTerm( #Foot Clearance is causing it to fail... hmmm figure this out in the morning
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "std": 0.005,
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names = FEET),
        },
        )
    # Keeps actions from going into runaway
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.01
        )

@configclass
class ThimbleTerminations(TerminationsCfg):   # keeps time_out and base_contact
    joint_state_invalid = DoneTerm(
        func=mdp.joint_state_invalid,
        params={"max_joint_vel": 1.0e6, "log_every": 24, "big_action_sq": 100.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0} # rad, ~57°
    )


@configclass
class ThimbleRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ThimbleRewards = ThimbleRewards()
    terminations: ThimbleTerminations = ThimbleTerminations()
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
            #Limits joint_toruqes to the legs only
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        # limit the joint TARGETS (absolute rad, after scale + default offset) to the URDF limits
        self.actions.joint_pos.clip = {
            "scapula_left": (-0.524, 0.524),
            "scapula_right": (-0.524, 0.524),
            "hip_left": (-0.436, 0.436),
            "hip_right": (-0.436, 0.436),
            "roll_to_pitch_left": (-1.571, 0), #changed roll_to_pitch limits to match true bounds.
            "roll_to_pitch_right": (0, 1.571),
            "knee_left": (-1.571, 1.571),
            "knee_right": (-1.571, 1.571),
            "ankle_left": (-1.571, 1.571),
            "ankle_right": (-1.571, 1.571),
        }
        # limit what the policy sees of its own last action, so large actions can't keep feeding on themselves
        self.observations.policy.actions.clip = (-5.0, 5.0)

        # terminations: falling = base touches the ground
        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE]

        # Safety clipping for any terms that aren't already clipped
        #for group in (self.observations.policy, getattr(self.observations, "critic", None)):
        #    if group is None:
        #        continue
        #    for term in group.__dict__.values():
        #        if isinstance(term, ObsTerm):
        #            term.clip = (-100.0, 100.0)

        # events: the defaults are sized for 30+ kg robots
        self.events.add_base_mass = None
        self.events.base_com = None   # re-enable later, scaled to Thimble, bc normally for bigger robots
        self.events.push_robot = EventTerm(
            func = mdp.push_by_setting_velocity,
            mode = "interval",
            interval_range_s = (4.0, 8.0),
            params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
        )
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE]

        # resets sized for Thimble
        # start velocity: Isaac Lab default is ±0.5 m/s and ±0.5 rad/s on every axis
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "z": (-0.25, 0.25),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.25, 0.25),
        }

        # start pose: add ±0.1 rad to each leg joint's default
        # (the default reset multiplies the default by 0.5–1.5, which never moves joints whose default is 0)
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            },
        )

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

@configclass
class ThimbleFlatEnvCfg_ROBUST(ThimbleFlatEnvCfg_PLAY):
    """Evaluation only: the original harsh Isaac Lab resets."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.events.push_robot = None
        self.events.reset_base.params["velocity_range"] = {
            k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        )

@configclass
class ThimbleFlatEnvCfg_ROBUST_ORIG(ThimbleFlatEnvCfg_ROBUST):
    """Original training conditions: harsh resets, no bad_orientation to compare with earlier runs."""
    def __post_init__(self):
        super().__post_init__()
        self.terminations.bad_orientation = None

@configclass
class ThimbleFlatEnvCfg_ORIG(ThimbleFlatEnvCfg):
    """Training conditions of the original run (2026-11-12): harsh Isaac Lab resets,
    no bad_orientation, no pushes. The logging-only diag termination stays."""
    def __post_init__(self):
        super().__post_init__()
        # harsh resets (Isaac Lab defaults)
        self.events.reset_base.params["velocity_range"] = {
            k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale, mode="reset",
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        )
        # the original run had neither of these
        self.terminations.bad_orientation = None
        self.events.push_robot = None
