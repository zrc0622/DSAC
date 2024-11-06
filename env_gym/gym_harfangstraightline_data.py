import gym
from harfang_env import register_envs  # 确保注册代码在运行前被执行
from harfang_env import dogfight_client as df

def env_creator(**kwargs):
    # connect to harfang
    df.connect("192.168.193.226", 11111)
    df.disable_log()
    df.set_renderless_mode(True)
    df.set_client_update_mode(True)
    try:
        env = gym.make('HarfangEnv-straightline')
        return env
    except:
        raise ModuleNotFoundError(
            "Warning: create harfang error"
        )
    
