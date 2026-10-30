from isaaclab.app import AppLauncher

# start Isaac Sim first, headless
app = AppLauncher(headless=True).app

# only now import Isaac Lab environment code
import isaaclab
print("isaaclab from:", isaaclab.__file__)

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
print("ok")

app.close()