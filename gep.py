import os
import sys
import numpy as np
import gym
import pickle
import argparse
import tensorflow as tf
from controllers import NNController
from representers import CheetahRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *

SAVING_FOLDER = './results/'
TRIAL_ID = 1
NB_RUNS = 1
ENV_ID = 'MountainCarContinuous-v0'
render = False
print_score = False
nb_play = 1 # number of episode to test each policy
NB_EPISODES = 100

def run_experiment(env_id, trial, nb_episodes, saving_folder):

    # create data path
    data_path = create_data_path(saving_folder, env_id, trial)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # GEP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # get GEP config
    if env_id=='MountainCarContinuous-v0':
        nb_bootstrap, nb_tests, nb_timesteps, offline_eval, controller, representer, policy_sampling_method,  \
        nb_rep, engineer_goal, goal_space, initial_space,  knn, noise, nb_weights, best_policy_selection = cmc_config()

    # overun some settings
    nb_episodes = int(nb_episodes/nb_play)
    nb_bootstrap = int(nb_bootstrap/nb_play)
    nb_tests = 100
    offline_eval = (1e6, 10) #(x,y): y evaluation episodes every x (done offline)

    train_perfs = []
    eval_perfs = []
    final_eval_perfs = []

    # compute test indices:
    test_ind = range(int(offline_eval[0])-1, nb_episodes, int(offline_eval[0]))

    # define environment
    env = gym.make(env_id)
    nb_act = env.action_space.shape[0]
    nb_obs = env.observation_space.shape[0]
    nb_rew = 1
    action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
    observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
    reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)
    policies = np.array([]).reshape(0, nb_weights)

    # bootstrap phase
    # # # # # # # # # # #
    for ep in range(nb_bootstrap):
        print('Bootstrap episode #', ep+1)
        # sample policy at random
        policy = sample_random_policy(nb_weights, mode=policy_sampling_method)

        # play policy and update knn
        obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                    representer, knn, render, nb_play)

        # save
        action_seqs, observation_seqs, reward_seqs, policies, train_perfs = save_data(action_seqs, observation_seqs,
                                                                                      reward_seqs, policies, train_perfs,
                                                                                      act, obs, rew, policy, print_score)



        # offline tests
        if ep in test_ind:
            best_policy = find_best_policy(select=best_policy_selection, knn=knn, engineer_goal=engineer_goal,
                                           train_perfs=train_perfs, nb_play=nb_play, policies=policies)
            offline_evaluations(offline_eval[1], best_policy, nb_rew, nb_timesteps, env, controller, eval_perfs)

    # exploration phase
    # # # # # # # # # # # #
    for ep in range(nb_bootstrap, nb_episodes):
        print('Random Goal episode #', ep+1)

        # random goal strategy
        policy = random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights)

        # play policy and update knn
        obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                    representer, knn, render, nb_play)

        # save
        action_seqs, observation_seqs, reward_seqs, policies, train_perfs = save_data(action_seqs, observation_seqs,
                                                                                      reward_seqs, policies, train_perfs,
                                                                                      act, obs, rew, policy, print_score)

        # offline tests
        if ep in test_ind:
            best_policy = find_best_policy(select=best_policy_selection, knn=knn, engineer_goal=engineer_goal,
                                           train_perfs=train_perfs, nb_play=nb_play, policies=policies)
            offline_evaluations(offline_eval[1], best_policy, nb_rew, nb_timesteps, env, controller, eval_perfs)

    # final evaluation phase
    # # # # # # # # # # # # # # #
    for ep in range(nb_tests):
        print('Test episode #', ep+1)
        best_policy = find_best_policy(select=best_policy_selection, knn=knn, engineer_goal=engineer_goal,
                                       train_perfs=train_perfs, nb_play=nb_play, policies=policies)
        offline_evaluations(1, best_policy, nb_rew, nb_timesteps, env, controller, final_eval_perfs)

    print('Final performance for the run: ', np.array(final_eval_perfs).mean())

    # wrap up and save
    # # # # # # # # # # #
    gep_memory = dict()
    gep_memory['actions'] = action_seqs.swapaxes(1, 2)
    gep_memory['observations'] = observation_seqs.swapaxes(1, 2)
    gep_memory['rewards'] = reward_seqs.swapaxes(1, 2)
    gep_memory['best_policy'] = best_policy
    gep_memory['train_perfs'] = np.array(train_perfs)
    gep_memory['eval_perfs'] = np.array(eval_perfs)
    gep_memory['final_eval_perfs'] = np.array(final_eval_perfs)
    gep_memory['representations'] = knn._X
    gep_memory['policies'] = knn._Y

    with open(data_path+'save_gep.pk', 'wb') as f:
        pickle.dump(gep_memory, f)


def find_best_policy(select='goal', train_perfs=None, nb_play=1, engineer_goal=None, knn=None, policies=None):

    if select == 'goal':
        best_policy = knn.predict(engineer_goal)[0, :]
    elif select == 'perf':
        perfs = np.array(train_perfs)
        perfs_grouped = []
        for i in range(0, perfs.shape[0], nb_play):
            perfs_grouped.append(perfs[i:i+nb_play].mean())
        ind_best = np.argmax(perfs_grouped)
        best_policy = policies[ind_best, :]

    return best_policy

def sample_random_policy(nb_weights, mode='uniform'):

    if mode == 'normal':
        return np.random.normal(0, 1, nb_weights)
    else:
        return np.random.random(nb_weights) * 2 - 1

def save_data(action_seqs, observation_seqs, reward_seqs, policies, train_perfs, act, obs, rew, policy, print_score):

    action_seqs = np.concatenate([action_seqs, act], axis=0)
    observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
    reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
    policies = np.concatenate([policies, policy.reshape(1,-1)], axis=0)
    rew = np.copy(rew[:, 0, :])
    nb_play = rew.shape[0]
    perfs = np.nansum(rew, axis=1)
    for i in range(nb_play):
        train_perfs.append(perfs[i])
        if print_score:
            print(np.nansum(perfs[i]))

    return action_seqs, observation_seqs, reward_seqs, policies, train_perfs

def play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller, representer, knn, render, nb_play=1):
    """
    Play a policy in the environment for a given number of timesteps, usin a NN controller.
    Then represent the trajectory and update the inverse model.
    """
    obs = np.zeros([nb_play, nb_obs, nb_timesteps + 1])
    act = np.zeros([nb_play, nb_act, nb_timesteps])
    rew = np.zeros([nb_play, nb_rew, nb_timesteps + 1])
    obs.fill(np.nan)
    act.fill(np.nan)
    rew.fill(np.nan)
    for p in range(nb_play):
        obs[p, :, 0] = env.reset()
        rew[p, :, 0] = 0
        done = False  # termination signal
        controller.reset(policy)
        for t in range(nb_timesteps):
            if done:
                break
            act[p, :, t] = controller.step(obs[p, :, t]).reshape(1, -1)
            out = env.step(np.copy(act[0, :, t]))
            if render:
                env.render()
            # env.render()
            obs[p, :, t + 1] = out[0]
            rew[p, :, t + 1] = out[1]
            done = out[2]

        # convert the trajectory into a representation (=behavioral descriptor)
        rep = representer.represent(obs[p, :, :], act[p, :, :])

        # update inverse model
        knn.update(X=rep, Y=policy)

    return obs, act, rew

def offline_evaluations(nb_eps, best_policy,  nb_rew, nb_timesteps, env, controller, eval_perfs):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    controller.reset(best_policy)

    returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)
        obs = env.reset()
        rew[:, 0] = 0
        done = False
        for t in range(nb_timesteps):
            if done:
                break
            act = controller.step(obs).reshape(1, -1)
            out = env.step(np.copy(act))
            # env.render()
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
        returns.append(np.nansum(rew))
    eval_perfs.append(np.array(returns).mean())

    return best_policy


def random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights):
    """
    Draw a goal, find policy associated to its nearest neighbor in the representation space, add noise to it.
    """
    # draw goal in goal space
    goal = np.copy(sample(goal_space))
    # scale goal to [-1,1]^N
    goal = scale_vec(goal, initial_space)

    # find policy of nearest neighbor
    policy = knn.predict(goal)[0]

    # add exploration noise
    policy += np.random.normal(0, noise*2, nb_weights) # noise is scaled by space measure
    policy_out = np.clip(policy, -1, 1)

    return policy_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', type=int, default=TRIAL_ID)
    parser.add_argument('--nb_episodes', type=int, default=NB_EPISODES)
    parser.add_argument('--env_id', type=str, default=ENV_ID)
    parser.add_argument('--saving_folder', type=str, default=SAVING_FOLDER)
    args = vars(parser.parse_args())

    gep_perf = np.zeros([NB_RUNS])
    for i in range(NB_RUNS):
        gep_perf[i] = run_experiment(**args)
        print(gep_perf)
        print('Average performance: ', gep_perf.mean())






