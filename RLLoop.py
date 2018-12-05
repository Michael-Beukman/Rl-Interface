import gym
from gym import wrappers
import datetime
import os
from math import log10, ceil
import numpy as np
from collections import deque
from copy import deepcopy

class RLLoop:
    def __init__(self, env, monitor_dir, model_name='', frequency=100, model_description=''):
        self.env = env
        self.frequency = frequency
        self.model_name = model_name
        if model_name == '':
            model_dir = 'General Models'
        else:
            model_dir = model_name
        if model_description == '':
            model_description = 'No description provided'

        self.model_description = model_description
        self.monitor_dir = os.path.join(monitor_dir, model_dir)

    def run_episodes(self, env, num_episodes, agent, verbose=False):
        """

        :param env: open Ai gym environment
        :param num_episodes: Number of episodes to run
        :param agent: A reinforcement learning agent. It has to have at least the methods specified in the readme
        :param verbose: Boolean indicating if the rewards should be outputted
        :return: a tuple of (episode_rewards, agent)
        """
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
            episode_rewards.append(episode_reward)
            if verbose:
                if episode % self.frequency == 0:
                    print("Average reward over {} episodes at episode {} : {}"
                          .format(self.frequency, episode, sum(episode_rewards[episode-self.frequency:])/self.frequency))
            agent.post_episode()
        return episode_rewards, agent

    @staticmethod
    def get_monitor_folder_name(self):
        date = datetime.datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        time_str = date.strftime('%Hh%Mm%Ss')

        return os.path.join(date_str + " " + time_str)

    def do_multiple_runs(self, num_runs, agent,num_episodes, doMonitor=True, seeding=True, verbose=False):
        """

        :param num_runs: Number of different runs to do
        :param agent: A reinforcement learning agent. Refer to the readme for the methods the agent needs to have
        :param num_episodes: Number of episodes per run
        :param doMonitor: Should the data be saved using openAI gym monitor
        :param seeding: Should the env be seeded with the run number (0, 1, ..., num_runs-1)
        :param verbose: Should the rewards be printed to the screen
        :return: tuple of (all_episode_rewards, deque of 10 newest agents)
        """
        monitor_folder_name = os.path.join(self.monitor_dir, self.get_monitor_folder_name(self))
        self.write_description_file(monitor_folder_name, num_runs, num_episodes)
        num_zeros = int(ceil(log10(num_runs)))
        all_episode_rewards = []
        newest_agents = deque(maxlen=10)
        for run_num in range(num_runs):
            agent.pre_run()
            folder = os.path.join(monitor_folder_name, str(run_num).zfill(num_zeros))
            if doMonitor:
                monitor = wrappers.Monitor(self.env, folder, video_callable=False)
            else:
                monitor = self.env
            if seeding:
                monitor.seed(run_num)
            episode_rewards, new_agent = self.run_episodes(monitor, num_episodes, deepcopy(agent), verbose=verbose)
            newest_agents.append(new_agent)
            all_episode_rewards.append(episode_rewards)

        return all_episode_rewards, newest_agents

    @staticmethod
    def analyse_rewards(episode_rewards):
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


    def write_description_file(self, folder, num_runs, num_episodes):
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = 'details.txt'
        delim = '='*100 + "\n"
        with open(os.path.join(folder, name), 'w+') as f:
            f.write('RLLoop Details file\n')
            f.write(delim)
            f.write("Model Details\n" + delim)
            f.write(self.model_description+'\n'+delim)
            f.write('\nRun Details\n' + delim)
            f.write("Number of runs: {}\n".format(num_runs))
            f.write("Number of episodes per run: {}\n".format(num_episodes))


