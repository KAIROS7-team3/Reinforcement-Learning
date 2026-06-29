"""OpenDrawer task registration."""

import gymnasium as gym

_PPO_CFG = "tool_transfer_bot.agents.ppo_cfg.open_drawer_ppo_cfg:OpenDrawerPPORunnerCfg"

gym.register(
    id="Isaac-OpenDrawer-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:OpenDrawerTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDrawer-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:OpenDrawerTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDrawer-Teacher-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:OpenDrawerTeacherEnvCfg_DEMO",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDrawer-Teacher-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:OpenDrawerTeacherEnvCfg_TELEOP",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)
