# mdp/terminations.py  (Thimble task)
#
# Termination that resets envs whose joint state is non-finite or physically
# impossible, plus lightweight diagnostics printed once per log interval.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def _absmax(x: torch.Tensor) -> torch.Tensor:
    """Max |x| over the whole tensor. NaN -> 0, but +/-inf is kept so it shows up in the log."""
    return torch.nan_to_num(x, nan=0.0, posinf=float("inf"), neginf=float("-inf")).abs().max()


class joint_state_invalid(ManagerTermBase):
    """Terminate envs whose joint positions/velocities are non-finite or exceed `max_joint_vel`.

    Isaac Lab's built-in `joint_vel_out_of_manual_limit` uses `abs(joint_vel) > max_velocity`,
    which is False for NaN, so a NaN env would never be reset. This term checks finiteness explicitly.

    Diagnostics: every `log_every` env steps, prints the running max of |joint_vel|,
    |raw action|, |policy obs| across all envs and how many env-steps were flagged,
    then resets those running maxima. With num_steps_per_env = 24, log_every = 24
    gives roughly one line per training iteration.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._device = env.device
        self._reset_stats()

    def _reset_stats(self):
        self._max_jv = torch.zeros((), device=self._device)
        self._max_act = torch.zeros((), device=self._device)
        self._max_obs = torch.zeros((), device=self._device)
        self._n_bad = torch.zeros((), dtype=torch.long, device=self._device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_joint_vel: float,
        log_every: int = 24,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        jp = asset.data.joint_pos[:, asset_cfg.joint_ids]
        jv = asset.data.joint_vel[:, asset_cfg.joint_ids]

        finite = torch.isfinite(jp).all(dim=1) & torch.isfinite(jv).all(dim=1)
        too_fast = jv.abs().amax(dim=1) > max_joint_vel  # NaN compares False here; `finite` catches it
        bad = ~finite | too_fast

        # ---- diagnostics: stay on the GPU, one host sync per log interval ----
        self._max_jv = torch.maximum(self._max_jv, _absmax(jv))
        self._max_act = torch.maximum(self._max_act, _absmax(env.action_manager.action))
        obs = getattr(env, "obs_buf", None)  # observations the policy acted on this step
        if isinstance(obs, dict) and isinstance(obs.get("policy"), torch.Tensor):
            self._max_obs = torch.maximum(self._max_obs, _absmax(obs["policy"]))
        self._n_bad += bad.sum()

        if log_every > 0 and env.common_step_counter % log_every == 0:
            print(
                f"[joint_state_invalid] step {env.common_step_counter}: "
                f"max|joint_vel| {self._max_jv.item():.3g}  "
                f"max|action| {self._max_act.item():.3g}  "
                f"max|obs| {self._max_obs.item():.3g}  "
                f"flagged {self._n_bad.item()}"
            )
            self._reset_stats()

        return bad