import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym
# from configs import cheetah_config

def create_data_path(saving_folder, env_id, trial_id):
    data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    if os.path.exists(data_path):
        i = 1
        while os.path.exists(saving_folder + env_id + '/' + str(trial_id + 100 * i) + '/'):
            i += 1
        trial_id += i * 100
        print('result_path already exist, trial_id changed to: ', trial_id)
    data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    os.mkdir(data_path)

    return data_path

def scale_vec(vector, initial_space):
    """
    Scale vector from initial space to [-1,1]^N
    """
    vec_in = np.copy(vector)
    vec_out = (vec_in - initial_space[:, 0]) * 2 / np.diff(initial_space).squeeze() - 1

    return vec_out

def sample(space):
    """
    Uniform sampling of a vector from the input space.
    Space must be a 2d-array such that 1st column are the mins and 2nd the max for each dimension
    """
    vec_out = np.random.random(space.shape[0]) * np.diff(space).squeeze() + space[:, 0]

    return vec_out

