from gym.envs.registration import register

# 注册环境
register(
    id='HarfangEnv-straightline',
    entry_point='harfang_env.HarfangEnv_GYM:HarfangEnv',
)

register(
    id='HarfangEnv-serpentine',
    entry_point='harfang_env.HarfangEnv_GYM:HarfangSerpentineEnv',
)

register(
    id='HarfangEnv-circular',
    entry_point='harfang_env.HarfangEnv_GYM:HarfangCircularEnv',
)