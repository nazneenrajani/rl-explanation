import math
import random
import inspect
import os
import time
from argparse import ArgumentParser
from itertools import count
from types import SimpleNamespace
from gym.spaces import Box
from gym_minigrid import *
from gym_minigrid.wrappers import *
from PIL import Image

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.distributions import Categorical, kl_divergence
from tensorboardX import SummaryWriter
from utils import *
from envs import *
import yaml

from model import *
from predicates import *
from general import *
from replay import *
import wandb
from constants import *

np.set_printoptions(suppress=True, precision=2)


def prediction_update_wandb(prediction, frisbee):
    wandb.log({'standard_Q/maxQ': torch.max(prediction.main_demon).cpu().data.numpy(),
               'standard_Q/minQ': torch.min(prediction.main_demon).cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done, commit=False)

    for i, predicate in enumerate(frisbee.predicates):
        wandb.log({f'{predicate.name()}_Q/maxQ': torch.max(prediction.prediction_demons[:, i]).cpu().data.numpy(),
                   f'{predicate.name()}_Q/minQ': torch.min(prediction.prediction_demons[:, i]).cpu().data.numpy()},
                  step=frisbee.trackers.training.steps_done, commit=False)


def optimize_dist_model(frisbee):
    if not frisbee.n_predicates > 0:
        return

    # --------------------------------------------------------------------
    # Sample from replay memory
    # --------------------------------------------------------------------
    if len(frisbee.memory) < frisbee.config.dist_batch_size:
        return

    transitions = frisbee.memory.sample(frisbee.config.dist_batch_size, 1.)
    batch = PrioritizedPredicateTransition(*zip(*transitions))

    # Construct the batches
    state_batch = torch.tensor(batch.state, dtype=torch.float32, device=frisbee.device)

    # Perform distillation using the target network
    if frisbee.config.distill_from_target_net:
        prediction = frisbee.target_net(state_batch)
        context = prediction.embedding.detach()
    else:
        prediction = frisbee.policy_net(state_batch)
        context = prediction.embedding.detach() if frisbee.config.detach_attn_context else prediction.embedding

    # Distillation as regularized regression
    dist_prediction = frisbee.dist_net(context,
                                       prediction.main_demon.detach(),
                                       prediction.prediction_demons.detach())

    if frisbee.config.regression_loss_fn == 'smooth_l1':
        regression_loss = F.smooth_l1_loss(dist_prediction.output.reshape(prediction.main_demon.shape),
                                           prediction.main_demon)
    elif frisbee.config.regression_loss_fn == 'l1':
        regression_loss = F.l1_loss(dist_prediction.output.reshape(prediction.main_demon.shape),
                                    prediction.main_demon)
    elif frisbee.config.regression_loss_fn == 'l2':
        regression_loss = F.mse_loss(dist_prediction.output.reshape(prediction.main_demon.shape),
                                     prediction.main_demon)
    else:
        raise NotImplementedError

    if frisbee.config.regularization == 'l1':
        regularization = frisbee.config.regularization_coef * dist_prediction.weights.abs().sum(1).mean()
    elif frisbee.config.regularization == 'entropy':
        if not frisbee.config.attn_softmax:
            raise AssertionError('Entropy regularization only works with softmax attention weights.')
        regularization = frisbee.config.regularization_coef * (- dist_prediction.weights *
                                                               (dist_prediction.weights + 1e-8).log()).sum(1).mean()
    else:
        raise NotImplementedError

    residual_prediction = frisbee.dist_net.forward_residuals(context.detach())
    if residual_prediction:
        residuals = dist_prediction.output.reshape(prediction.main_demon.shape) - prediction.main_demon
        residual_loss = F.mse_loss(residual_prediction.residuals.reshape(prediction.main_demon.shape), residuals)
        wandb.log({'loss/residuals': wandb.Histogram(residuals.cpu().data.numpy().flatten())},
                  step=frisbee.trackers.training.steps_done, commit=False)
    else:
        residual_loss = torch.tensor(0.)

    loss = regression_loss + regularization

    # Wandb logging for the losses
    wandb.log({'loss/regression_loss': regression_loss.cpu().data.numpy(),
               'loss/regularization': regularization.cpu().data.numpy(),
               'loss/residual_loss': residual_loss.cpu().data.numpy(),
               'loss/total_regression_loss': loss.cpu().data.numpy(),
               'attention/weights': wandb.Histogram(dist_prediction.weights.cpu().data.numpy().flatten())},
              step=frisbee.trackers.training.steps_done, commit=False)

    loss += residual_loss

    # Optimize the model
    frisbee.dist_optimizer.zero_grad()
    loss.backward()
    for param in frisbee.dist_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    frisbee.dist_optimizer.step()


def optimize_horde_model(frisbee):
    # --------------------------------------------------------------------
    # Sample from replay memory
    # --------------------------------------------------------------------
    if len(frisbee.memory) < frisbee.config.batch_size:
        return

    transitions = frisbee.memory.sample(frisbee.config.batch_size,
                                        (frisbee.config.replay_buffer_beta_end -
                                         frisbee.config.replay_buffer_beta_start)/
                                        frisbee.config.replay_buffer_beta_decay
                                        + frisbee.config.replay_buffer_beta_start)

    batch = PrioritizedPredicateTransition(*zip(*transitions))

    # Construct the batches
    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=frisbee.device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=frisbee.device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=frisbee.device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=frisbee.device)
    predicate_rewards_batch = torch.tensor(batch.predicate_rewards, dtype=torch.float32, device=frisbee.device)
    done_batch = torch.tensor(batch.done, dtype=torch.uint8, device=frisbee.device)
    indices_batch = batch.index
    weights_batch = torch.tensor(batch.weight, dtype=torch.float32, device=frisbee.device).unsqueeze(1)

    # wandb.log({'step/replay_rewards': wandb.Histogram(batch.reward)},
    #           step=frisbee.trackers.training.steps_done, commit=False)
    # wandb.log({'step/replay_indices': wandb.Histogram(batch.index)},
    #           step=frisbee.trackers.training.steps_done, commit=False)
    # wandb.log({'step/replay_dones': wandb.Histogram(batch.done)},
    #           step=frisbee.trackers.training.steps_done, commit=False)
    # wandb.log({'step/replay_states': wandb.Histogram(state_batch.data.numpy().flatten())},
    #           step=frisbee.trackers.training.steps_done, commit=False)

    # Find the non-terminal states
    non_final_mask = 1 - done_batch
    non_final_indices = np.nonzero(non_final_mask).squeeze()
    non_final_next_states = next_state_batch[non_final_indices]

    # --------------------------------------------------------------------
    # Predict using the main Q/policy network
    # --------------------------------------------------------------------

    policy_net_prediction = frisbee.policy_net(state_batch)

    # Gather Q values at the actions taken by the agent
    main_demon_at_actions = policy_net_prediction.main_demon.gather(1, action_batch)

    if frisbee.n_predicates > 0:
        prediction_demons_at_actions = policy_net_prediction.prediction_demons.\
            gather(2, action_batch.unsqueeze(1).expand(action_batch.shape[0], frisbee.n_predicates, action_batch.shape[1]))

        control_demons_at_actions = policy_net_prediction.control_demons.gather(2, action_batch.unsqueeze(1).
                                                                                expand(action_batch.shape[0],
                                                                                       len(frisbee.config.predicates),
                                                                                       action_batch.shape[1]))

    # Wandb logging
    if frisbee.trackers.training.steps_done % 256 == 0:
        prediction_update_wandb(policy_net_prediction, frisbee)

    # --------------------------------------------------------------------
    # Predict using the target network (at non-final next states)
    # --------------------------------------------------------------------

    target_net_prediction = frisbee.target_net(non_final_next_states)

    # Calculate the value function at s'
    v_next = torch.zeros(frisbee.config.batch_size, device=frisbee.device)

    if frisbee.config.q_target == 'standard':
        v_next[non_final_mask] = target_net_prediction.main_demon.max(1)[0].detach()
    elif frisbee.config.q_target == 'double':
        policy_net_prediction_q = frisbee.policy_net.forward_q(non_final_next_states)
        v_next[non_final_mask] = target_net_prediction.main_demon[range(non_final_next_states.shape[0]),
                                                                  policy_net_prediction_q.argmax(1)].detach()
    else:
        raise NotImplementedError

    if frisbee.n_predicates > 0:
        # Calculate the auxiliary value functions at s'
        aux_vs_next = torch.zeros((frisbee.config.batch_size, len(frisbee.predicates)), device=frisbee.device)

        # Evaluation operator with greedy policy from main Q
        if frisbee.config.q_target == 'standard':
            aux_vs_next[non_final_mask] = target_net_prediction.prediction_demons.detach().\
                gather(dim=-1, index=target_net_prediction.main_demon.argmax(1).detach().unsqueeze(1).
                       expand(target_net_prediction.prediction_demons.shape[:-1]).unsqueeze(2)).squeeze(2)
        elif frisbee.config.q_target == 'double':
            aux_vs_next[non_final_mask] = target_net_prediction.prediction_demons.detach(). \
                gather(dim=-1, index=policy_net_prediction_q.argmax(1).detach().unsqueeze(1).
                       expand(target_net_prediction.prediction_demons.shape[:-1]).unsqueeze(2)).squeeze(2)

        # Off-policy optimality operator

    # --------------------------------------------------------------------
    # Compute the losses
    # --------------------------------------------------------------------

    # Compute the targets for the Bellman updates
    target_q_at_actions = (v_next * frisbee.config.gamma) + reward_batch
    if frisbee.n_predicates > 0:
        target_aux_qs_at_actions = (aux_vs_next * frisbee.config.gamma) + predicate_rewards_batch

    # Calculate the Q-learning loss
    q_loss = (F.smooth_l1_loss(main_demon_at_actions, target_q_at_actions.unsqueeze(1), reduction='none')
              * weights_batch).mean()
    # q_loss = F.smooth_l1_loss(main_demon_at_actions, target_q_at_actions.unsqueeze(1))
    if frisbee.n_predicates > 0:
        aux_qs_loss = (F.smooth_l1_loss(prediction_demons_at_actions,
                                        target_aux_qs_at_actions.unsqueeze(2), reduction='none')
                       * weights_batch.unsqueeze(1)).mean()
        # aux_qs_loss = F.smooth_l1_loss(prediction_demons_at_actions, target_aux_qs_at_actions.unsqueeze(2))
    else:
        aux_qs_loss = torch.tensor(0.)

    # Update the replay memory priorities
    abs_td_errors = torch.abs(main_demon_at_actions.squeeze() - target_q_at_actions).cpu().data.numpy() + 1e-3
    frisbee.memory.update_priorities(indices_batch, abs_td_errors)

    # Total loss
    loss = q_loss + aux_qs_loss

    # Wandb logging for the losses
    wandb.log({'loss/q_loss': q_loss.cpu().data.numpy(),
               'loss/aux_qs_loss': aux_qs_loss.cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done, commit=False)

    # Optimize the model
    frisbee.optimizer.zero_grad()
    loss.backward()
    for param in frisbee.policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    frisbee.optimizer.step()


def train(frisbee):

    state = frisbee.env.reset()
    seeds = np.array(frisbee.env.get_seeds())

    # Episodic trackers
    frisbee.trackers.episode.durations = np.zeros(frisbee.config.n_envs)
    frisbee.trackers.episode.returns = np.zeros(frisbee.config.n_envs)
    frisbee.trackers.episode.predicate_returns = np.zeros((frisbee.config.n_envs, frisbee.n_predicates))

    wandb.log({'episode/seeds': wandb.Histogram(seeds)},
              step=frisbee.trackers.training.steps_done, commit=False)

    # Main learning loop
    for _ in count(0, frisbee.config.n_envs):

        # Sample an action to perform
        action = select_epsilon_greedy_action(state, frisbee)

        # Take a step: single action in the environment
        next_state, reward, done, info = frisbee.env.step(action.data.numpy())
        next_state = next_state

        # Extract the predicate rewards
        predicate_rewards = [e['predicates'] for e in info]

        # Extract the termination info
        terminal = [e['terminal'] for e in info]

        # Store the transitions in memory
        for e in zip(state, action, next_state, reward, predicate_rewards, terminal):
            frisbee.memory.push(*e)

        # Store the actions into the experience
        frisbee.experience.add_parallel_actions(action, done, seeds)

        # Move to the next state
        state = next_state

        # Update the episodic trackers
        frisbee.trackers.episode.durations += 1
        frisbee.trackers.episode.returns += np.array(reward)
        frisbee.trackers.episode.predicate_returns += np.array(predicate_rewards)

        # Optimize the model
        if frisbee.trackers.training.steps_done % frisbee.config.update_freq == 0 and \
                frisbee.trackers.training.steps_done > frisbee.config.bootstrap:
            optimize_horde_model(frisbee)
            for _ in range(frisbee.config.dist_steps):
                optimize_dist_model(frisbee)

        # Update the target network
        update_target_network(frisbee)

        # Checkpoint the model every so often
        checkpoint_horde(frisbee)

        # Check if we're done with any episode
        if np.any(done):
            # Reset the state and update the seeds
            indices = np.where(done)[0]
            state[indices] = frisbee.env.reset_subset(indices)
            seeds[indices] = frisbee.env.get_seeds(indices)

            wandb.log({'episode/seeds': wandb.Histogram(seeds)},
                      step=frisbee.trackers.training.steps_done, commit=False)
            # --------------------------------------------------------------------
            # Things we do after every episode
            # --------------------------------------------------------------------
            # Update the training trackers
            frisbee.trackers.training.episodes_done += len(indices)
            frisbee.trackers.training.best_return = max(frisbee.trackers.training.best_return,
                                                        *frisbee.trackers.episode.returns[indices])

            # Update wandb with the episode trackers
            for idx in indices:
                wandb.log({f'episode/returns': frisbee.trackers.episode.returns[idx],
                           f'episode/durations': frisbee.trackers.episode.durations[idx]},
                          step=frisbee.trackers.training.steps_done + idx, commit=False)
                wandb.log({f'episode/{predicate.name()}_returns': frisbee.trackers.episode.predicate_returns[idx][i]
                           for i, predicate in enumerate(frisbee.predicates)},
                          step=frisbee.trackers.training.steps_done + idx)

            # Reset tracking for these environments
            frisbee.trackers.episode.durations[indices] = 0.
            frisbee.trackers.episode.returns[indices] = 0.
            frisbee.trackers.episode.predicate_returns[indices] = 0.

        # Update the training trackers
        frisbee.trackers.training.steps_done += frisbee.config.n_envs

        # Update wandb
        step_update_training_horde_wandb(frisbee)

        if frisbee.trackers.training.steps_done > frisbee.config.num_steps:
            break

    frisbee.env.close()
    checkpoint_horde(frisbee, force=True)


def main(args):
    # Create a frisbee -- a SimpleNamespace to toss around that contains general information
    frisbee = create_training_frisbee(args)

    # Set up the predicates and add them to the frisbee
    add_sns_to_frisbee(setup_predicates(frisbee), frisbee)

    # Set up the environment and add it to the frisbee
    add_sns_to_frisbee(setup_horde_env(frisbee), frisbee)

    # Set up the model and add it to the frisbee
    add_sns_to_frisbee(setup_horde_model(frisbee), frisbee)

    # Set up the training trackers and add them to the frisbee
    add_sns_to_frisbee(setup_training_trackers(frisbee), frisbee)

    # Do an initial update to weights and biases
    watch_model_wandb(frisbee)

    # Train
    train(frisbee)


if __name__ == '__main__':
    # Set up the argument parser
    parser = ArgumentParser('Pass in arguments!')
    parser.add_argument('--config', '-c', help='Path to the config .yaml file.', type=str, required=True)
    args = parser.parse_args()

    # Run!
    main(args)
