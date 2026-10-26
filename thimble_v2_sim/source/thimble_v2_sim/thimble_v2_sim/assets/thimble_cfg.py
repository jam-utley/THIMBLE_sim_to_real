"""Thimble v1 (Open Duck Mini v2-based biped) articulation config for Isaac Lab 2.3.x.

Pattern copied from isaaclab_assets/robots/cassie.py (Isaac Lab v2.3.2).

EVERY value marked PLACEHOLDER must be replaced with:
  * joint / body names from YOUR converted USD (open it in Isaac Sim and read the prim tree), and
  * actuator numbers from the HX-35HM datasheet + your own servo system-identification.
The sim will happily train on wrong numbers; the real robot will not walk on them.
"""

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

THIMBLE_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

THIMBLE_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Put the converted USD (from MJCF/URDF) here: source/thimble_v1/thimble_v1/assets/data/
        usd_path=os.path.join(THIMBLE_ASSETS_DIR, "thimble_v2.usd"),
        activate_contact_sensors=True,  # required by the contact-sensor rewards/terminations
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),  # trunk height (m) in your nominal standing pose, plus a few mm
        #Set initial joint position
        joint_pos={ 
            "scapula_left"       :   0.0,
            "hip_left"           :   0.0,
            "roll_to_pitch_left" :   0.0,#math.radians(30.0),
            "knee_left"          :   0.0,#math.radians(90.0),
            "ankle_left"         :   0.0,
            "scapula_right"      :   0.0,
            "hip_right"          :   0.0,
            "roll_to_pitch_right":   0.0,
            "knee_right"         :   0.0,
            "ankle_right"        :   0.0,
            "head_yaw"           :   0.0,
            "head_roll"          :   0.0,
            "head_pitch"         :   0.0,#math.radians(15.0),
            "neck_c7_vertebra"   :   0.0,#math.radians(20.0)
        },
        joint_vel={".*": 0.0}, # ".*" is a regex pattern meaning "every joint" so here it is "every joint starts at 0.0 velocity" which is what we want
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # HX-35HM servos run their own position loop. ImplicitActuatorCfg is only a starting point:
        # once you have step-response data, compare it with the explicit/delayed PD models in
        # isaaclab.actuators (or an actuator network, Hwangbo et al. 2019, arXiv:1901.08652).
        "left_leg": DCMotorCfg(
            joint_names_expr = ["scapula_left", "hip_left", "roll_to_pitch_left", "knee_left", "ankle_left"],
            saturation_effort=3.43,    # stall torque (N·m): the top of the torque-speed line
            velocity_limit=5.5,        # no-load speed (rad/s): where available torque reaches zero
            effort_limit=3.43,         # absolute torque cap; lower this if you want a continuous-torque limit
            stiffness=10.0,            # PLACEHOLDER: same meaning as before (N·m/rad)
            damping=0.5,               # PLACEHOLDER
            armature=0.01,             # PLACEHOLDER: nonzero, see below
        ),
        "right_leg": DCMotorCfg(
            joint_names_expr = ["scapula_right", "hip_right", "roll_to_pitch_right", "knee_right", "ankle_right"],
            saturation_effort=3.43,    # stall torque (N·m): the top of the torque-speed line
            velocity_limit=5.5,        # no-load speed (rad/s): where available torque reaches zero
            effort_limit=3.43,         # absolute torque cap; lower this if you want a continuous-torque limit
            stiffness=10.0,            # PLACEHOLDER: same meaning as before (N·m/rad)
            damping=0.5,               # PLACEHOLDER
            armature=0.01,             # PLACEHOLDER: nonzero, see below
        ),
        "neck_head": ImplicitActuatorCfg(
            joint_names_expr = ["head_yaw", "head_roll", "head_pitch", "neck_c7_vertebra", "antenna_left", "antenna_right"],
            effort_limit_sim = 3.43,
            velocity_limit_sim = 5.5,
            stiffness = 10.0, #PLACEHOLDER
            damping = 0.5,    #PLACEHOLDER
            armature = 0.5,   #PLACEHOLDER
            )
    },
)
"""Thimble v1 with a single implicit-PD actuator group (to be replaced after servo SysID)."""
