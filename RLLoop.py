import gym
from gym import wrappers
import datetime
import os
from math import log10, ceil
import numpy as np


class RLLoop:
    def __init__(self, env, monitor_dir):
        self.env = env
        self.monitor_dir = monitor_dir

    @staticmethod
    def run_episodes(env, num_episodes, agent, verbose=False):
        episode_rewards = []
        for episode in range(1, num_episodes+1):
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.pre_step(state)
                next_state, reward, done, _ = env.step(action)
                agent.post_step(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            if verbose:
                print("Reward at episode {} : {}".format(episode, episode_reward))
            episode_rewards.append(episode_reward)
            agent.post_episode()
        return episode_rewards, agent

    @staticmethod
    def get_monitor_folder_name(self):
        date = datetime.datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        time_str = date.strftime('%Hh%Mm%Ss')

        return os.path.join(date_str + " " + time_str)

    def do_multiple_runs(self, num_runs, agent,num_episodes, doMonitor=True, seeding=True, verbose=False):
        monitor_folder_name = os.path.join(self.monitor_dir, self.get_monitor_folder_name(self))
        num_zeros = int(ceil(log10(num_runs)))
        all_episode_rewards = []
        for run_num in range(num_runs):
            folder = os.path.join(monitor_folder_name, str(run_num).zfill(num_zeros))
            if doMonitor:
                monitor = wrappers.Monitor(self.env, folder, video_callable=False)
            else:
                monitor = self.env
            if seeding:
                monitor.seed(run_num)
            episode_rewards, _ = self.run_episodes(monitor, num_episodes, agent, verbose=verbose)
            all_episode_rewards.append(episode_rewards)

        return all_episode_rewards

    @staticmethod
    def analyse_rewards(self, episode_rewards):
        rewards = np.array(episode_rewards)

        average_per_run = np.average(rewards, axis=1)
        std_per_run = np.std(rewards, axis=1)

        averaged_results = np.average(rewards, axis=0)
        averaged_std = np.std(rewards, axis=0)
        total_average = np.average(averaged_results)

        dic = {"averagePerRun":average_per_run, "stdPerRun":std_per_run,
               "averagedResults":averaged_results, "averagedStd":averaged_std,
               "totalAverage":total_average}

        return dic


