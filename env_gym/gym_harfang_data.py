import gym
from harfang_env import register_envs  # 确保注册代码在运行前被执行
from harfang_env import dogfight_client as df

def env_creator(**kwargs):
    # connect to harfang
    df.connect("10.249.242.57", 11111)
    df.disable_log()
    df.set_renderless_mode(True)
    df.set_client_update_mode(True)
    try:
        return gym.make('HarfangEnv-v0')
    except:
        raise ModuleNotFoundError(
            "Warning: create harfang error"
        )
    
