from isaaclab.app import AppLauncher

app = AppLauncher(headless=False).app

import gymnasium as gym
import torch
import thimble_v2_sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Thimble-Velocity-Flat-v0"
cfg = parse_env_cfg(TASK, num_envs=1)
cfg.terminations.base_contact = None  # keep running even if it falls over
env = gym.make(TASK, cfg=cfg)

robot = env.unwrapped.scene["robot"]
term = env.unwrapped.action_manager.get_term("joint_pos")
knee_slots = [term._joint_names.index(n) for n in ("knee_left", "knee_right")]
knee_ids, _ = robot.find_joints(["knee_left", "knee_right"])

env.reset()
actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
for step in range(300):
    a = 1.0 if (step // 100) % 2 == 0 else -1.0  # hold +1, then -1, then +1 (2 s each)
    actions[:, knee_slots] = a
    env.step(actions)
    if step % 10 == 0:
        q = robot.data.joint_pos[0, knee_ids].tolist()
        print(f"step {step:3d}  action {a:+.0f}  knee {q[0]:+.3f} {q[1]:+.3f}")

env.close()
app.close()