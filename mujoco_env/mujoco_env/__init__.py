from gymnasium.envs.registration import register

register(
    id="mujoco_env/FrankaReach-v0",
    entry_point="mujoco_env.envs:FrankaReachingEnv",
)