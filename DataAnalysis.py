import matplotlib.pyplot as plt
import os
import glob
import json
import numpy as np


def get_run_folder(directory):
    to_search = os.path.join(directory, '*')
    return max(glob.glob(to_search), key=os.path.getctime)


def get_stats(folder):
    stats_file = glob.glob(os.path.join(folder, '*.stats.json'))
    if len(stats_file) ==0:
        raise Exception('No *.stats.json files in folder')
    stats_file = stats_file[0]

    json_data = json.load(open(stats_file, 'rb'))
    return json_data


def get_all_runs_stats(parent_dir, just_rewards=True):
    all_folders = [os.path.join(parent_dir, o) for o in os.listdir(parent_dir)
                   if os.path.isdir(os.path.join(parent_dir, o))]
    all_data = []
    for run_folder in all_folders:
        all_data.append(get_stats(run_folder))
    if not just_rewards:
        return all_data
    all_rewards = [i['episode_rewards'] for i in all_data]
    return np.array(all_rewards)


def analyse_rewards(rewards):
    mean_rewards = np.mean(rewards, axis=0)
    overall_mean = np.mean(rewards)
    std_rewards = np.std(rewards, axis=0)
    max_rewards = np.max(rewards, axis=0)
    return {"mean_rewards": mean_rewards, "overall_mean": overall_mean,
            "std_rewards": std_rewards, "max_rewards":max_rewards}


def plot_data(points, std=None):
    # print(points.shape)
    # plt.errorbar([i for i in range(len(points))], points, yerr=std, fmt='o')
    # plt.show()
    plt.plot(points)
    plt.show()


def get_summary(reward_stats):
    details = analyse_rewards(reward_stats)
    s = "="*100 + '\n'
    for key in details:
        s += "{}: {}\n".format(key, details[key])
    s += "="*100
    return s


def do_all(parent_dir, run_folder=None):
    if run_folder is None:
        run_folder = get_run_folder(parent_dir)
    all_run_stats = get_all_runs_stats(run_folder)
    mean_rewards = analyse_rewards(all_run_stats)
    plot_data(mean_rewards['mean_rewards'], mean_rewards['std_rewards'])
    print(get_solve_stats(950, all_run_stats))
    print(get_summary(all_run_stats))

def get_solve_stats(solve_reward, run_stats, num_to_average=100):
    solve_eps = []
    for run in run_stats:
        for i in range(100, len(run)):
            ave = np.mean(run[i-100:i])
            if ave >= solve_reward:
                solve_eps.append(i)
                break
        else:
            solve_eps.append(len(run))

    solve_eps = np.array(solve_eps)
    maxi = np.amax(solve_eps)
    mini = np.amin(solve_eps)
    ave = np.mean(solve_eps)
    return {"Min steps to solve": mini, "Max steps to solve": maxi, "Average steps to solve":ave}

if __name__ == '__main__':
    # f_name = '/Users/michaelbeukman/Research/An Empirical Analysis of Policy Parameterisations for Low-Dimensional Continuous Control Tasks/Tasks/Task 002 - Continuous Control/monitorData/Continuous Reinforce with bias/2018-12-03 09h56m41s'
    f_name = '/Users/michaelbeukman/Research/An Empirical Analysis of Policy Parameterisations for Low-Dimensional Continuous Control Tasks/Tasks/Task 002 - Continuous Control/monitorData/Continuous Reinforce with bias/2018-12-02 18h29m59s'
    do_all('yee', run_folder=f_name)