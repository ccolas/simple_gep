import numpy as np

from controllers import NNController
from representers import CheetahRepresenter, CMCRepresenter, PendRepresenter
from inverse_models import KNNRegressor

from gep_utils import *



def cmc_config():

    # run parameters
    nb_bootstrap = 10
    nb_tests = 100
    nb_timesteps = 1000
    offline_eval = (1e6, 10)  # (x,y): y evaluation episodes every x (done offline)

    # controller parameters
    hidden_sizes = []
    dim_state = 2
    dim_act = 1
    controller_tmp = 1.
    activation = 'relu'
    scale = np.array([[-1.2,0.6],[-0.07,0.07]])
    controller = NNController(dim_state, hidden_sizes, dim_act, controller_tmp, scale, activation)
    nb_weights = controller.nb_weights

    # representer
    representer = CMCRepresenter()
    initial_space = representer.initial_space
    goal_space = representer.initial_space  # space in which goal are sampled
    nb_rep = representer.dim
    engineer_goal = np.array([0.5, 1.5, 5.])  # engineer goal
    # scale engineer goal to [-1,1]^N
    engineer_goal = scale_vec(engineer_goal, initial_space)
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = 0.1

    # policy sampling method
    # 'uniform': in [-1, 1]
    # 'normal' (0, 1)
    policy_sampling_method = 'uniform'


    # best policy selection
    # to find best policy in archive,  use:
    # 'goal': the closest to some engineered goal in the behavioral space,
    # 'perf': the policy that obtained the highest performance
    best_policy_selection = 'goal'

    return nb_bootstrap, nb_tests, nb_timesteps, offline_eval, controller, representer, \
           policy_sampling_method, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights, best_policy_selection

