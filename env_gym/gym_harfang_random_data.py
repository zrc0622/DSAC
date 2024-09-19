import gym
from harfang_env import register_envs  # 确保注册代码在运行前被执行
from harfang_env import dogfight_client as df

def env_creator(**kwargs):
    # connect to harfang
    port = kwargs["env_port"]
    df.connect("172.27.58.131", port)
    df.disable_log()
    df.set_renderless_mode(True)
    df.set_client_update_mode(True)
    try:
        return gym.make('HarfangEnv-r-v0')
    except:
        raise ModuleNotFoundError(
            "Warning: create harfang error"
        )
    
