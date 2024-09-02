from gym.envs.registration import register

# 注册环境
register(
    id='HarfangEnv-v0',
    entry_point='harfang_env.HarfangEnv_GYM:HarfangEnv',
    max_episode_steps=6000,
)
