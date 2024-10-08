import numpy as np
import torch
import os

from utils.initialization import create_env
from utils.common_utils import set_seed
from utils.plot import plot_dif2, draw_pos



class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update(
            {"reward_scale": None, "repeat_num": None}
        )  # evaluation don't need to scale reward
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs["is_render"]
        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.plot_folder = kwargs["plot_folder"]
        os.makedirs(self.plot_folder, exist_ok=True)
        self.eval_save = kwargs.get("eval_save", False)

        self.print_time = 0
        self.print_iteration = -1

        self.harfang_env = kwargs["harfang_env"]
        self.env_id = kwargs["env_id"]

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        # harfang
        success = 0
        fire_success = 0

        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)

            obs = next_obs
            info = next_info

            # harfang
            if self.harfang_env:
                if info['success']: success = 1
                if info['fire_success']: fire_success = 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        if self.harfang_env:
            return episode_return, success, fire_success    
        else: 
            return episode_return
        
    def run_an_episode_with_plot(self, iteration, render=True):
        # harfang
        success = 0
        fire_success = 0

        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False

        dif=[]
        fire=[]
        lock=[]
        missile=[]
        self_pos = []
        oppo_pos = []
        step = -1

        while not (done or info["TimeLimit.truncated"]):
            step += 1
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info, if_fire, before_state, after_state, locked, step_success = self.env.step_test(action)
            
            dif.append(self.env.loc_diff)
            if if_fire:
                fire.append(step)
            if locked:
                lock.append(step)
            if before_state:
                missile.append(step)
            self_pos.append(self.env.get_pos())
            oppo_pos.append(self.env.get_oppo_pos())

            obs_list.append(obs)
            action_list.append(action)

            obs = next_obs
            info = next_info

            # harfang
            if self.harfang_env:
                if info['success']: success = 1
                if info['fire_success']: fire_success = 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        
        draw_pos(f'pos_{iteration}.png', self_pos, oppo_pos, fire, lock, self.plot_folder) 
        plot_dif2(dif, lock, missile, fire, self.plot_folder, f'dif_{iteration}.png')

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        if self.harfang_env:
            return episode_return, success, fire_success    
        else: 
            return episode_return

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        episode_success_list = []
        episode_fire_success_list = []
        if self.harfang_env:
            for i in range(n):
                if i == n-1:
                    episode_return, success, fire_success = self.run_an_episode_with_plot(iteration, self.render)
                else:
                    episode_return, success, fire_success = self.run_an_episode(iteration, self.render)
                episode_return_list.append(episode_return)
                episode_success_list.append(success)
                episode_fire_success_list.append(fire_success)
                if self.env_id == 'gym_harfang_infinite':
                    print('t_f: ', self.env.total_fire, 't_s: ', self.env.total_success)
            return np.mean(episode_return_list), np.mean(episode_success_list), np.mean(episode_fire_success_list)
        else: 
            for _ in range(n):
                episode_return_list.append(self.run_an_episode(iteration, self.render))
            return np.mean(episode_return_list)

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)


def create_evaluator(**kwargs):
    evaluator = Evaluator(**kwargs)
    print("Create evaluator successfully!")
    return evaluator
