import gym
from harfang_env import register_envs  # 确保注册代码在运行前被执行
from harfang_env import dogfight_client as df

if __name__ == "__main__":
    # connect to harfang
    df.connect("172.30.58.126", 11111)
    df.disable_log()
    df.set_renderless_mode(False)
    df.set_client_update_mode(True)

    env = gym.make('HarfangEnv-v0')
    state = env.reset()

    for _ in range(10000):
        action = env.action_space.sample()  # 这里随机采样动作，您可以替换为智能体的动作
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
