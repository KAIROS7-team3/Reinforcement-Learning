"""ReturnTool task registration."""

import gymnasium as gym

_PPO_CFG = "tool_transfer_bot.agents.ppo_cfg.return_tool_ppo_cfg:ReturnToolPPORunnerCfg"

gym.register(
    id="Isaac-ReturnTool-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:ReturnToolTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ReturnTool-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:ReturnToolTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ReturnTool-Teacher-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:ReturnToolTeacherEnvCfg_DEMO",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ReturnTool-Teacher-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:ReturnToolTeacherEnvCfg_TELEOP",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
    disable_env_checker=True,
)
