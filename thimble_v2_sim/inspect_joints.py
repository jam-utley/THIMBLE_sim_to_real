from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import thimble_v2_sim.tasks  # noqa: F401  (registers your tasks)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Thimble-Velocity-Flat-v0"
env = gym.make(TASK, cfg=parse_env_cfg(TASK, num_envs=1))
robot = env.unwrapped.scene["robot"]

limits = robot.data.joint_pos_limits[0]
default = robot.data.default_joint_pos[0]
print(f"\n{'joint':28s} {'lower':>8s} {'upper':>8s} {'default':>8s}")
for i, name in enumerate(robot.joint_names):
    lo, hi, d = limits[i, 0].item(), limits[i, 1].item(), default[i].item()
    flag = "  <-- default OUTSIDE limits" if not (lo <= d <= hi) else ""
    flag += "  <-- range ~0, joint locked" if hi - lo < 1e-3 else ""
    print(f"{name:28s} {lo:+8.3f} {hi:+8.3f} {d:+8.3f}{flag}")

term = env.unwrapped.action_manager.get_term("joint_pos")
print("\nJoints the policy drives:", term._joint_names)

env.close()
app.close()