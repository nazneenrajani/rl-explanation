import gym
import gym_minigrid
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.register import register
import geomloss
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import ot
from scipy.stats import spearmanr, pearsonr

from envs import MiniGridRewardWrapper
import ml_metrics as mlm

import gym

from stable_baselines.common.policies import FeedForwardPolicy, CnnPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, ACKTR, SAC, DQN
import imageio
from mutual_info import *

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from scipy.stats import rankdata

import pickle

sequential_colors = sns.color_palette("RdPu", 10)
# Set the palette to the "pastel" default palette:
sns.set_palette("RdPu")


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[16, 8],
                                           feature_extraction="mlp")


class MiniGridBinaryRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super(MiniGridBinaryRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward > 0.:
            return 1.0
        return 0.


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class GridExample(MiniGridEnv):

    def __init__(self,
                 agent_at=0,
                 goal_at=2,
                 lava_at=None,
                 wall_at=None,
                 max_steps=100,
                 height=4,
                 width=5
                 ):
        positions = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
        self.agent_start_pos = positions[agent_at]

        self.agent_start_dir = 0

        self.goal_pos = positions[goal_at] if goal_at is not None else None
        self.lava_pos = positions[lava_at] if lava_at is not None else None
        self.wall_pos = positions[wall_at] if wall_at is not None else None

        super().__init__(
            height=height,
            width=width,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.goal_pos:
            # Place a goal square in the bottom-right corner
            self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())
        if self.lava_pos:
            # Place a goal square in the bottom-right corner
            self.grid.set(self.lava_pos[0], self.lava_pos[1], Lava())
        if self.wall_pos:
            # Place a goal square in the bottom-right corner
            self.grid.set(self.wall_pos[0], self.wall_pos[1], Wall())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class MiniGridForwardBackwardUpDownActionsOnly(gym.Wrapper):

    def __init__(self, env):
        super(MiniGridForwardBackwardUpDownActionsOnly, self).__init__(env)

    def step(self, action):
        step_count_before = self.env.step_count
        # Action 0 is forward, action 1 is backward, action 2 is up, action 3 is right
        if action == 0 or action == 4:
            # Actual mini-grid action is 2
            sp, rew, done, info = self.env.step(2)

        elif action == 1 or action == 5:
            # 2 x right, 1 x forward, 2 x right
            _, r_1, d_1, _ = self.env.step(1)
            _, r_2, d_2, _ = self.env.step(1)
            _, r_3, d_3, _ = self.env.step(2)
            _, r_4, d_4, _ = self.env.step(1)
            sp, r_5, d_5, info = self.env.step(1)

            rew = r_1 + r_2 + r_3 + r_4 + r_5
            done = max(d_1, d_2, d_3, d_4, d_5)

        elif action == 2 or action == 6:
            # 1 x left, 1 x forward, 1 x right
            _, r_1, d_1, _ = self.env.step(0)
            _, r_2, d_2, _ = self.env.step(2)
            sp, r_3, d_3, info = self.env.step(1)

            rew = r_1 + r_2 + r_3
            done = max(d_1, d_2, d_3)

        elif action == 3 or action == 7:
            # 1 x right, 1 x forward, 1 x left
            _, r_1, d_1, _ = self.env.step(1)
            _, r_2, d_2, _ = self.env.step(2)
            sp, r_3, d_3, info = self.env.step(0)

            rew = r_1 + r_2 + r_3
            done = max(d_1, d_2, d_3)

        else:
            raise NotImplementedError

        self.env.step_count = step_count_before + 1
        return sp, rew, done, info


class MiniGridNoisyTransitions(gym.Wrapper):

    def __init__(self, env, noise_level):
        super(MiniGridNoisyTransitions, self).__init__(env)
        self.obs = None
        self.noise_level = noise_level

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.obs

    def step(self, action):
        coin = np.random.rand()
        if coin >= self.noise_level:
            self.obs, rew, done, info = self.env.step(action)
            return self.obs, rew, done, info
        else:
            return self.obs, 0., False, {}


def plot_matrix(matrix, figsize, xticklabels, yticklabels, xlabel, ylabel, save_path, vmin=None, vmax=None, mask=None):
    plt.figure(figsize=figsize)
    p = sns.heatmap(matrix, vmin=vmin, vmax=vmax,
                    mask=mask,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    linewidths=0, linecolor='white',
                    annot=np.round(matrix.astype(np.float), 1), square=True, cbar=False, cmap="Blues")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def make_matrix_blocks(matrix, block_sizes):
    assert (matrix.shape[0] == matrix.shape[1] == np.sum(block_sizes))

    def insert_plus(mat, idx):
        n = mat.shape[0]
        mat = np.insert(mat, idx, np.zeros(n) * np.nan, axis=1)
        return np.insert(mat, idx, np.zeros(n + 1) * np.nan, axis=0)

    for i, insertion_pt in enumerate(np.array(block_sizes[:-1]).cumsum()):
        matrix = insert_plus(matrix, insertion_pt + i)

    mask = np.isnan(matrix)
    return matrix, mask


def run():
    noise_level = 0.0
    agent_at = 0
    goal_at = 2
    lava_at = None
    wall_at = None

    def wrap(env, noise):
        return MiniGridNoisyTransitions(MiniGridForwardBackwardUpDownActionsOnly(FullyObsWrapper(env)), noise)

    def test_policy(model, env, n_trajs=10):
        returns = []
        for rep in range(n_trajs):
            ret = 0
            obs = env.reset()
            for step in range(1000):
                obs, rew, done, info = env.step(model.predict(obs)[0])
                ret += rew
                if done:
                    break
            returns.append(ret)
        return np.mean(returns), np.std(returns)

    def collect_trajectories(model, env, n_trajs=10):
        trajectories = []
        for rep in range(n_trajs):
            trajectories.append([])
            obs = env.reset()
            for step in range(1000):
                action = model.predict(obs)[0]
                next_obs, rew, done, info = env.step(action)
                trajectories[-1].append((obs, action, rew))
                obs = next_obs
                if done:
                    break
            trajectories[-1].append((obs, None, None))
        return trajectories

    def compute_return(trajectories):
        returns = []
        for trajectory in trajectories:
            ret = 0
            for _, _, rew in trajectory[:-1]:
                ret += rew
            returns.append(ret)
        return np.mean(returns), np.std(returns)

    def make_gif(model, env, path):
        images = []
        obs = env.reset()
        img = env.render(mode='rgb_array',
                         highlight=False)
        for i in range(1000):
            images.append(img)
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            img = env.render(mode='rgb_array',
                             highlight=False)
            if done:
                break
        images.append(img)

        imageio.mimsave(path, images, fps=1)

    def make_env(setting, noise, rank, seed=0):
        def _init():
            env = wrap(GridExample(*setting), noise)
            env.seed(seed + rank)
            return env

        return _init

    num_cpu = 4  # Number of processes to use

    # Environment variants
    env_settings = [(0, 2, None, None),
                    (0, 5, None, None),
                    (1, 4, None, None),
                    (2, 0, None, None),
                    (5, 0, None, None),
                    (4, 1, None, None),
                    (0, 2, 1, None),
                    (0, 5, 1, None),
                    (0, 5, 2, None),
                    (0, 5, 4, None),
                    (0, 2, None, 1),
                    (0, 5, None, 1),
                    (0, 5, None, 2),
                    (0, 5, None, 4),
                    (0, 5, 2, 3),
                    (0, 5, 3, 2),
                    (0, None, 2, None),
                    (2, None, 0, None)]

    env_blocks = ['move agent & goal',
                  'add lava',
                  'add wall',
                  'add lava & wall',
                  'remove goal']

    block_sizes = [6, 4, 4, 2, 2]

    models = []

    predicates = []

    algorithm = [PPO2, DQN, SAC, ACKTR][0]
    postfix = ['ppo', 'dqn', 'sac', 'acktr'][0]

    try:
        evaluation_matrix, distances = pickle.load(open('fat/exp1/eval_data', 'rb'))
    except FileNotFoundError:

        # Go over each of the environments
        for i, setting in enumerate(env_settings):
            models.append([])
            for j in range(5):
                # Store renderings of the source environment
                env = wrap(GridExample(*setting), noise_level)
                env.reset()
                rendering = env.render(mode='rgb_array', highlight=False)
                im = Image.fromarray(rendering)
                draw = ImageDraw.Draw(im)
                im.save(f"fat/grids/render_{i + 1}_clean.png")
                font = ImageFont.truetype("/Library/Fonts/Futura.ttc", 20)
                draw.text((3, 3), f"{i+1}", (255, 255, 255), font=font)

                im.save(f"fat/grids/render_{i+1}.png")
                # plt.imsave(f"fat/grids/render_{'.'.join([str(e) for e in setting])}.png", env.render(mode='rgb_array',
                #                                                                                     highlight=False))
                # plt.imsave(f"fat/grids/render_{i+1}.png", env.render(mode='rgb_array', highlight=False))

                # Create and wrap the vectorized environment
                if algorithm != SAC:
                    env = SubprocVecEnv([make_env(setting, noise_level, i) for i in range(num_cpu)])
                else:
                    env = DummyVecEnv([make_env(setting, noise_level, 0)])
                model = algorithm(CustomPolicy, env, verbose=1)
                try:
                    # Loading the model
                    model = algorithm.load(f"fat/exp1/{'.'.join([str(e) for e in setting]) + '.' + str(noise_level) + '.' + str(j)}.{postfix}")
                except ValueError:
                    # Training the model
                    model.learn(total_timesteps=500000)
                    model.save(f"fat/exp1/{'.'.join([str(e) for e in setting]) + '.' + str(noise_level) + '.' + str(j)}.{postfix}")

                models[-1].append(model)

        ####### Evaluation ########


        evaluation_matrix = []
        distances = []
        n_trajs = 50

        trajectories_across_settings = []
        scenarios_across_settings = []
        for i, setting in enumerate(env_settings):
            # Collect some trajectories from the task by running the policy trained on the task
            env = MiniGridBinaryRewardWrapper(wrap(GridExample(*setting, max_steps=20), noise_level))
            scenario = env.reset()
            scenarios_across_settings.append(scenario)
            trajectories = collect_trajectories(models[i][0], env, n_trajs) # all the models trained to be optimal
            trajectories_across_settings.append(trajectories)
            # Make a gif of what this policy has learned to do
            make_gif(models[i][0], env, f"fat/exp1/gif_{'.'.join([str(e) for e in setting])}.gif")

        for i, setting in enumerate(env_settings):
            # Go over the target environments
            evaluation_matrix.append([])
            distances.append([])
            # Grab the trajectories and scenario for the source task
            source_trajectories, source_scenario = trajectories_across_settings[i], scenarios_across_settings[i]
            for j, other_setting in enumerate(env_settings):
                # Create the target environment
                other_env = MiniGridBinaryRewardWrapper(wrap(GridExample(*other_setting, max_steps=20), noise_level))

                # Grab the trajectories and scenario for the target task
                target_trajectories, target_scenario = trajectories_across_settings[j], scenarios_across_settings[j]

                # Evaluate the return when you run the source task's policy on the target task
                return_means = []
                for k in range(5):
                    print (i, j, k)
                    return_mean, return_std = test_policy(models[i][k], other_env, n_trajs=n_trajs)
                    return_means.append(return_mean)

                return_mean = np.mean(return_means)

                evaluation_matrix[-1].append(return_mean)

                metrics = [metric_V_distance(source_trajectories, target_trajectories),
                           metric_edit_distance(source_scenario, target_scenario),
                           metric_edit_distance_scored(source_scenario, target_scenario, 5.),
                           metric_sinkhorn_distance(source_trajectories, target_trajectories),
                           metric_wasserstein_distance(source_trajectories, target_trajectories)]
                distances[-1].append(metrics)

        evaluation_matrix = np.array(evaluation_matrix)

    pickle.dump((evaluation_matrix, distances), open('fat/exp1/eval_data', 'wb'))

    evaluation_matrix_mod, mask = make_matrix_blocks(evaluation_matrix, block_sizes)

    locs = np.insert(np.array(block_sizes).cumsum(), 0, 0)
    locs = (locs[1:] - locs[:-1]) / 2. + locs[:-1] + np.arange(len(block_sizes))
    locs[-1] = min(locs[-1], evaluation_matrix_mod.shape[0] - 1)
    print(locs)

    labels = ['' for _ in range(len(evaluation_matrix_mod))]
    for e, i in zip(env_blocks, locs):
        labels[int(i)] = e

    plot_matrix(evaluation_matrix_mod,
                (10, 8),
                labels,
                labels,
                'Target Task', 'Source Task',
                f'fat/exp1/evaluation_matrix_{postfix}_mod.png', 0, 1, mask)

    plot_matrix(evaluation_matrix,
                (10, 8),
                ['.'.join([str(e) for e in s]) for s in env_settings],
                ['.'.join([str(e) for e in s]) for s in env_settings],
                'Target Task', 'Source Task',
                f'fat/exp1/evaluation_matrix_{postfix}.png', 0, 1)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(evaluation_matrix, vmin=0., vmax=1., xticklabels=['.'.join([str(e) for e in s]) for s in env_settings],
    #             yticklabels=['.'.join([str(e) for e in s]) for s in env_settings], linewidths=3, linecolor='black',
    #             annot=True, square=True, cbar=False)
    # plt.ylabel('Source Task')
    # plt.xlabel('Target Task')
    # plt.title(f'Return (averaged over {n_trajs} trials) [ticks: (agent_loc, goal_loc, lava_loc, wall_loc)]')
    # plt.tight_layout()
    # plt.savefig(f'fat/exp1/evaluation_matrix_{postfix}.png')
    # plt.close()

    success_matrix = (evaluation_matrix > 0.5)
    plot_matrix(success_matrix,
                (10, 8),
                ['.'.join([str(e) for e in s]) for s in env_settings],
                ['.'.join([str(e) for e in s]) for s in env_settings],
                'Target Task', 'Source Task',
                f'fat/exp1/success_matrix_{postfix}.png', 0, 1)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(success_matrix, vmin=0., vmax=1., xticklabels=['.'.join([str(e) for e in s]) for s in env_settings],
    #             yticklabels=['.'.join([str(e) for e in s]) for s in env_settings], linewidths=3, linecolor='black',
    #             annot=True, square=True, cbar=False)
    # plt.ylabel('Source Task')
    # plt.xlabel('Target Task')
    # plt.title(f'Success (return > 0.5 averaged over {n_trajs} trials) [ticks: (agent_loc, goal_loc, lava_loc, wall_loc)]')
    # plt.tight_layout()
    # plt.savefig(f'fat/exp1/success_matrix_{postfix}.png')
    # plt.close()

    distances = np.array(distances)
    print(distances.shape)
    print(distances)
    distances = distances.transpose(2, 0, 1)
    distance_types = ['Value', 'Edit', 'Scored Edit', 'Sinkhorn', 'Wasserstein']
    for i, dist_mat in enumerate(distances):
        dist_mat_mod, mask = make_matrix_blocks(dist_mat, block_sizes)

        plot_matrix(dist_mat_mod,
                    (10, 8),
                    labels,
                    labels,
                    'Target Task', 'Source Task',
                    f'fat/exp1/distance_matrix_{distance_types[i].lower()}_mod.png', mask=mask)

        plot_matrix(dist_mat,
                    (10, 8),
                    ['.'.join([str(e) for e in s]) for s in env_settings],
                    ['.'.join([str(e) for e in s]) for s in env_settings],
                    'Target Task', 'Source Task',
                    f'fat/exp1/distance_matrix_{distance_types[i].lower()}.png')

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(dist_mat,
        #             xticklabels=['.'.join([str(e) for e in s]) for s in env_settings],
        #             yticklabels=['.'.join([str(e) for e in s]) for s in env_settings],
        #             linewidths=3, linecolor='black',
        #             annot=True, square=True, cbar=False)
        # plt.ylabel('Source Task')
        # plt.xlabel('Target Task')
        # plt.title(f'{distance_types[i]} Distance [ticks: (agent_loc, goal_loc, lava_loc, wall_loc)]')
        # plt.tight_layout()
        # plt.savefig(f'fat/exp1/distance_matrix_{distance_types[i].lower()}.png')
        # plt.close()

        print(spearmanr(1 - evaluation_matrix.flatten(), dist_mat.flatten()),
              spearmanr(1 - success_matrix.flatten(), dist_mat.flatten()),
              pearsonr(1 - evaluation_matrix.flatten(), dist_mat.flatten()),
              pearsonr(1 - success_matrix.flatten(), dist_mat.flatten()),
              mutual_information([evaluation_matrix.flatten()[:, np.newaxis], dist_mat.flatten()[:, np.newaxis]]))

    correlations = []
    f1_scores = {e: [] for e in range(1, 11)}
    relevant_k = []
    for i, dist_mat_i in enumerate(distances):
        correlations.append([])
        for j, dist_mat_j in enumerate(distances):
            print (spearmanr(dist_mat_i.flatten(), dist_mat_j.flatten()),
                   pearsonr(dist_mat_i.flatten(), dist_mat_j.flatten()))
            correlations[-1].append(spearmanr(dist_mat_i.flatten(), dist_mat_j.flatten())[0])

        for e in range(1, 11):
            f1_scores[e].append([])

        for task in range(len(env_settings))[:-2]:
            print (i, task)

            generalizes = np.where(success_matrix[task])[0]
            if i == 0:
                relevant_k.append(len(generalizes))
            ranks = np.argsort(dist_mat_i[task])#rankdata(dist_mat_i[task], method='ordinal')
            print (generalizes, ranks)
            for k in range(1, 11):
                pre = precision(generalizes, ranks, k=k)
                rec = recall(generalizes, ranks, k=k)
                try:
                    f1 = 2 * pre * rec/ (pre + rec)
                except ZeroDivisionError:
                    f1 = 0.
                f1_scores[k][-1].append(f1)

    for k in range(1, 11):
        print (np.array(f1_scores[k]))
    print (relevant_k)




    plt.figure(figsize=(10, 8))
    sns.heatmap(np.array(correlations),
                xticklabels=distance_types,
                yticklabels=distance_types,
                linewidths=3, linecolor='black',
                annot=True, square=True, cbar=False)
    plt.ylabel('Distance Type')
    plt.xlabel('Distance Type')
    plt.title(f'Spearman correlation between distances')
    plt.tight_layout()
    plt.savefig(f'fat/exp1/dist_correlation_matrix.png')
    plt.close()


def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def metric_V_distance(source_trajectories, target_trajectories):
    # Compare the source and target task values
    def compute_return(trajectory):
        ret = 0
        gamma = 0.99
        for i, (_, _, rew) in enumerate(trajectory[:-1]):
            ret += rew * gamma ** i
        return ret

    V_source = np.mean([compute_return(e) for e in source_trajectories])
    V_target = np.mean([compute_return(e) for e in target_trajectories])

    return abs(V_source - V_target)


def metric_predicate_V_distance(source_trajectories, target_trajectories, predicate_fns):
    # Compare the source and target task predicate values
    def compute_predicate_return(trajectory, predicate_fn):
        ret = 0
        for (obs, act, _), (next_obs, _, _) in zip(trajectory[:-1], trajectory[1:]):
            ret += predicate_fn(obs, act, next_obs)
        return ret

    pred_V_source = np.array([compute_predicate_return(e, f) for f in predicate_fns
                              for e in source_trajectories]).reshape(len(predicate_fns), -1).mean(axis=1)
    pred_V_target = np.array([compute_predicate_return(e, f) for f in predicate_fns
                              for e in target_trajectories]).reshape(len(predicate_fns), -1).mean(axis=1)

    return np.sum(np.abs(pred_V_source - pred_V_target))


def metric_edit_distance(source_scenario, target_scenario):
    # Count the number of features which differ
    return np.sum(np.array(source_scenario) != np.array(target_scenario))


def metric_edit_distance_scored(source_scenario, target_scenario, scoring_function):
    # Count the weighted sum of features which differ
    return np.sum((np.array(source_scenario) != np.array(target_scenario)) * scoring_function)


def metric_sinkhorn_distance(source_trajectories, target_trajectories):
    source_data = np.array([e[0] for traj in source_trajectories for e in traj])
    target_data = np.array([e[0] for traj in target_trajectories for e in traj])

    # loss, x, y = sinkhorn_optimization(source_data, target_data)
    # return loss
    return ot_sinkhorn(source_data, target_data)


def metric_wasserstein_distance(source_trajectories, target_trajectories):
    source_data = np.array([e[0] for traj in source_trajectories for e in traj])
    target_data = np.array([e[0] for traj in target_trajectories for e in traj])

    # loss, x, y = sinkhorn_optimization(source_data, target_data, blur=0.001)
    # return loss
    return ot_wasserstein(source_data, target_data)


def ot_wasserstein(source_data, target_data):
    source_data = source_data.reshape(source_data.shape[0], -1)
    target_data = target_data.reshape(target_data.shape[0], -1)

    source_data, source_counts = np.unique(source_data, axis=0, return_counts=True)
    target_data, target_counts = np.unique(target_data, axis=0, return_counts=True)

    M = ot.dist(source_data, target_data, 'hamming')
    M /= M.max()

    source_dist, target_dist = source_counts / np.sum(source_counts), target_counts / np.sum(target_counts)
    distance = ot.emd2(source_dist, target_dist, M)
    return distance


def ot_sinkhorn(source_data, target_data):
    source_data = source_data.reshape(source_data.shape[0], -1)
    target_data = target_data.reshape(target_data.shape[0], -1)

    source_data, source_counts = np.unique(source_data, axis=0, return_counts=True)
    target_data, target_counts = np.unique(target_data, axis=0, return_counts=True)

    M = ot.dist(source_data, target_data, 'hamming')
    M /= M.max()

    source_dist, target_dist = source_counts / np.sum(source_counts), target_counts / np.sum(target_counts)
    distance = ot.sinkhorn2(source_dist, target_dist, M, 1e-2)
    return distance[0]


def sinkhorn_optimization(source_data, target_data, loss_type='sinkhorn', blur=0.05):
    loss = SamplesLoss(loss=loss_type, p=2, blur=blur)
    lr = 0.1
    epochs = 500

    source_data = torch.tensor(source_data.reshape(source_data.shape[0], -1), dtype=torch.float32)
    target_data = torch.tensor(target_data.reshape(target_data.shape[0], -1), dtype=torch.float32)

    # Make sure that we won't modify the reference samples
    x, y = source_data.clone(), target_data.clone()
    x.requires_grad = True

    for i in range(epochs):
        # Compute cost and gradient
        loss_value = loss(x, y)
        [g] = torch.autograd.grad(loss_value, [x])

        # in-place modification of the tensor's values
        x.data -= lr * len(x) * g

    return loss_value.data.numpy(), x, y


# def sinkhorn_optimization_with_learned_costs(source_data, target_data):
#     loss = SamplesLoss(loss='sinkhorn', p=2)
#     lr = 0.1
#     epochs = 500
#
#     source_data = torch.tensor(source_data.reshape(source_data.shape[0], -1), dtype=torch.float32)
#     target_data = torch.tensor(target_data.reshape(target_data.shape[0], -1), dtype=torch.float32)
#
#     # Make sure that we won't modify the reference samples
#     x, y = source_data.clone(), target_data.clone()
#     x.requires_grad = True
#
#     for i in range(epochs):
#         # Compute cost and gradient
#         loss_value = loss(x, y)
#         [g] = torch.autograd.grad(loss_value, [x])
#
#         # in-place modification of the tensor's values
#         x.data -= lr * len(x) * g
#
#     return loss_value, x, y


# if __name__ == '__main__':
#     run()

if __name__ == '__main__':
    setting = (2, 0, None, None)
    env = GridExample(height=4, width=5, *setting)
    env = MiniGridNoisyTransitions(MiniGridForwardBackwardUpDownActionsOnly(FullyObsWrapper(env)), 0.)

    env.reset()

    rendering = env.render(mode='rgb_array', highlight=False)
    im = Image.fromarray(rendering)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/Library/Fonts/Futura.ttc", 20)
    draw.text((3, 3), f"3", (255, 255, 255), font=font)
    im.save(f"fat/grids/render_3_replacement.png")

    # plt.imsave('fat/example_maze.png', env.render(mode='rgb_array', highlight=False))



# if __name__ == '__main__':
#     import torch
#
#     env1 = FullyObsWrapper(gym.make('MiniGrid-LavaCrossingS9N1-v0'))
#     env2 = FullyObsWrapper(gym.make('MiniGrid-LavaCrossingS9N2-v0'))
#     env3 = FullyObsWrapper(gym.make('MiniGrid-SimpleCrossingS9N1-v0'))
#
#     a1 = [2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2]
#     a2 = [1, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 2]
#     a3 = [2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2]
#
#     s1 = []
#     s2 = []
#     s3 = []
#
#     s1.append(env1.reset())
#     s2.append(env2.reset())
#     s3.append(env3.reset())
#
#     for a in a1:
#         s1.append(env1.step(a)[0])
#
#     for a in a2:
#         s2.append(env2.step(a)[0])
#
#     for a in a3:
#         s3.append(env3.step(a)[0])
#
#     s1, s2, s3 = np.array(s1), np.array(s2), np.array(s3)
#     print (s1.shape, s2.shape)
#
#     s1_flat, s2_flat = torch.tensor(s1.reshape(s1.shape[0], -1), dtype=torch.float32), torch.tensor(s2.reshape(s2.shape[0], -1), dtype=torch.float32)
#     s3_flat = torch.tensor(s3.reshape(s3.shape[0], -1), dtype=torch.float32)
#
#     print (s1_flat.shape, s2_flat.shape)
#
#     loss = SamplesLoss()
#
#     # print (loss.forward(s1_flat[-10:], s2_flat[-10:]))
#
#     # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
#     # dist, P, C = sinkhorn(s1_flat, s2_flat)
#
#
#     def gradient_descent(loss, lr=1):
#         """Flows along the gradient of the loss function.
#
#         Parameters:
#             loss ((x_i,y_j) -> torch float number):
#                 Real-valued loss function.
#             lr (float, default = 1):
#                 Learning rate, i.e. time step.
#         """
#
#         # Parameters for the gradient descent
#         Nsteps = 400
#
#         # Make sure that we won't modify the reference samples
#         x_i, y_j = s1_flat.clone(), s3_flat.clone()
#         option = 2
#         if option == 1:
#             transform = nn.Parameter(torch.randn(243, 2) * 0.01)
#         elif option == 2:
#             transform = torch.eye(243, 243)
#
#         # We're going to perform gradient descent on Loss(α, β)
#         # wrt. the positions x_i of the diracs masses that make up α:
#         x_i.requires_grad = True
#
#         for i in range(Nsteps):  # Euler scheme ===============
#             # Compute cost and gradient
#             if option == 1:
#                 loss_value = loss(torch.matmul(x_i, transform), torch.matmul(y_j, transform))
#                 [g] = torch.autograd.grad(loss_value, [transform])
#                 transform.data += 0.0001 * g
#                 # print (transform)
#             # Compute cost and gradient
#             loss_value = loss(torch.matmul(x_i, transform), torch.matmul(y_j, transform))
#             [g] = torch.autograd.grad(loss_value, [x_i])
#
#             # in-place modification of the tensor's values
#             x_i.data -= lr * len(x_i) * g
#
#         print (loss_value)
#
#         print (transform)
#         print (y_j)
#         print (x_i.shape)
#
#         plt.imshow(s1_flat.reshape(-1, 9, 9, 3).data.numpy().reshape(-1, 9, 3))
#         plt.show()
#
#         plt.imshow(x_i.reshape(-1, 9, 9, 3).data.numpy().reshape(-1, 9, 3))
#         plt.show()
#
#         if option == 1:
#             f0 = torch.matmul(s1_flat, transform).data.numpy()
#             f1 = torch.matmul(x_i, transform).data.numpy()
#             f2 = torch.matmul(y_j, transform).data.numpy()
#
#             plt.scatter(f0[:, 0], f0[:, 1], label='lava')
#             plt.scatter(f1[:,0], f1[:, 1], label='lava_t')
#             plt.scatter(f2[:, 0], f2[:, 1], label='no_lava')
#             plt.legend()
#             plt.show()
#
#     gradient_descent(loss, lr=0.1)


# sinkhorn = SinkhornDistance(eps=0.1, max_iter=10000, reduction=None)
# dist, P, C = sinkhorn(s1_flat, s2_flat)

# print (dist, P, C)
# import matplotlib.pyplot as plt
#
# plt.imshow(P)
# plt.colorbar()
# plt.show()
