from gym.envs.registration import register

# 注册环境
register(
    id='HarfangEnv-v0',
    entry_point='harfang_env.HarfangEnv_GYM:HarfangEnv',
    max_episode_steps=6000,
)

register(
    id='HarfangEnv-r-v0',
    entry_point='harfang_env.HarfangEnv_GYM:RandomHarfangEnv',
    max_episode_steps=6000,
)

register(
    id='HarfangEnv-i-v0',
    entry_point='harfang_env.HarfangEnv_GYM:InfiniteHarfangEnv',
    max_episode_steps=6000,
)