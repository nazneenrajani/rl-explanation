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
from replay import Transition, PredicateTransition, ReplayMemory
import wandb
from constants import *


# def torch_forward_kl(input, target):
#     # (batch size, outcomes)


def prediction_update_wandb(prediction, frisbee):
    wandb.log({'standard_Q/maxQ': torch.max(prediction.q).cpu().data.numpy(),
               'standard_Q/minQ': torch.min(prediction.q).cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done, commit=False)

    for i, predicate in enumerate(frisbee.predicates):
        wandb.log({f'{predicate.name()}_Q/maxQ': torch.max(prediction.aux_qs[:, i]).cpu().data.numpy(),
                   f'{predicate.name()}_Q/minQ': torch.min(prediction.aux_qs[:, i]).cpu().data.numpy()},
                  step=frisbee.trackers.training.steps_done, commit=False)

    if prediction.tilde_q:
        wandb.log({'tilde_Q/maxQ': torch.max(prediction.tilde_q).cpu().data.numpy(),
                   'tilde_Q/minQ': torch.min(prediction.tilde_q).cpu().data.numpy()},
                  step=frisbee.trackers.training.steps_done, commit=False)


def optimize_model(frisbee):
    # --------------------------------------------------------------------
    # Sample from replay memory
    # --------------------------------------------------------------------
    if len(frisbee.memory) < frisbee.config.batch_size:
        return

    transitions = frisbee.memory.sample(frisbee.config.batch_size)
    batch = PredicateTransition(*zip(*transitions))

    # Find the non-terminal states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=frisbee.device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Construct the batches
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    predicate_rewards_batch = torch.cat(batch.predicate_rewards, dim=0)

    # --------------------------------------------------------------------
    # Predict using the main Q/policy network
    # --------------------------------------------------------------------

    policy_net_prediction = frisbee.policy_net(state_batch)

    # Q values
    q_all_actions = policy_net_prediction.q
    # Gather Q values at the actions taken by the agent
    q_at_actions = policy_net_prediction.q.gather(1, action_batch)

    # Compute the epsilon greedy policy and detach to make sure no updates happen
    # Changed this to use the greedy policy computed from the target network instead!
    # pi = torch.tensor(frisbee.pi(q_all_actions, frisbee.trackers.training.steps_done).detach().unsqueeze(1),
    #                   device=frisbee.device, requires_grad=False)

    # Auxiliary Q values
    if frisbee.n_predicates > 0:
        aux_qs_all_actions = policy_net_prediction.aux_qs
        # Gather auxiliary Q values at the actions taken by the agent
        aux_qs_at_actions = policy_net_prediction.aux_qs.gather(2, action_batch.unsqueeze(1).
                                                                expand(action_batch.shape[0],
                                                                       len(frisbee.config.predicates),
                                                                       action_batch.shape[1]))

        # Attention weights
        attention_weights = policy_net_prediction.attention

        # Tilde Q values
        tilde_q_all_actions = policy_net_prediction.tilde_q
        # Gather tilde Q values at the actions taken by the agent
        tilde_q_at_actions = policy_net_prediction.tilde_q.gather(2, action_batch.unsqueeze(1).
                                                                  expand(action_batch.shape[0],
                                                                         frisbee.config.n_attention_heads,
                                                                         action_batch.shape[1]))

        # Compute the Boltzmann matching policy
        tilde_pi = frisbee.tilde_pi(tilde_q_all_actions, frisbee.trackers.training.steps_done).to(frisbee.device)

    # Wandb logging
    prediction_update_wandb(policy_net_prediction, frisbee)

    # --------------------------------------------------------------------
    # Predict using the target network (at non-final next states)
    # --------------------------------------------------------------------

    target_net_prediction = frisbee.target_net(non_final_next_states)

    # Calculate the value function at s'
    v_next = torch.zeros(frisbee.config.batch_size, device=frisbee.device)
    v_next[non_final_mask] = target_net_prediction.q.max(1)[0].detach()

    if frisbee.n_predicates > 0:
        # Calculate the auxiliary value functions at s'
        aux_vs_next = torch.zeros((frisbee.config.batch_size, len(frisbee.predicates)), device=frisbee.device)

        # Different choices of updates for the auxiliary Q functions
        # Optimality operator update
        # aux_vs_next[non_final_mask] = target_net_prediction.aux_qs.max(2)[0].detach()

        # Evaluation operator with mixing
        # target_pi = frisbee.pi(target_net_prediction.q,
        #                        frisbee.trackers.training.steps_done).detach().unsqueeze(2).to(frisbee.device)
        # aux_vs_next[non_final_mask] = torch.matmul(target_net_prediction.aux_qs.detach(), target_pi).squeeze(-1)

        # Evaluation operator with greedy policy from main Q
        aux_vs_next[non_final_mask] = target_net_prediction.aux_qs.detach().gather(dim=-1,
                                                                                   index=target_net_prediction.q.argmax(1).
                                                                                   detach().unsqueeze(1).
                                                                                   expand(target_net_prediction.aux_qs.
                                                                                          shape[:-1]).unsqueeze(2)).\
            squeeze(2)

        # --------------------------------------------------------------------
        # Predict using the target network (at current states)
        # This is used to find the target for policy matching
        # --------------------------------------------------------------------

        target_q = frisbee.target_net.forward_q(state_batch)
        # Use the greedy policy from the target network as the KL target rather than the epsilon-greedy policy
        # from the policy network
        pi = torch.zeros(target_q.shape).to(frisbee.device).scatter_(dim=1, index=target_q.argmax(1).unsqueeze(1), value=1)

    # --------------------------------------------------------------------
    # Compute the losses
    # --------------------------------------------------------------------

    # Compute the targets for the Bellman updates
    target_q_at_actions = (v_next * frisbee.config.gamma) + reward_batch
    if frisbee.n_predicates > 0:
        target_aux_qs_at_actions = (aux_vs_next * frisbee.config.gamma) + predicate_rewards_batch

    # Calculate the Q-learning loss
    q_loss = F.smooth_l1_loss(q_at_actions, target_q_at_actions.unsqueeze(1))
    if frisbee.n_predicates > 0:
        aux_qs_loss = F.smooth_l1_loss(aux_qs_at_actions, target_aux_qs_at_actions.unsqueeze(2))
    else:
        aux_qs_loss = torch.tensor(0.)

    if frisbee.n_predicates > 0:
        # Calculate the policy matching loss
        if frisbee.config.matching_loss == 'kl':
            # TODO: Fix for more than one attention head
            matching_loss = frisbee.config.matching_loss_coef * F.kl_div(input=(tilde_pi + 1e-5).log().squeeze(), target=pi,
                                                                         reduction='batchmean')
        elif frisbee.config.matching_loss == 'bce':
            matching_loss = frisbee.config.matching_loss_coef * F.binary_cross_entropy(input=tilde_pi, target=pi)
        elif frisbee.config.matching_loss == 'none':
            matching_loss = torch.tensor(0.)
        else:
            raise NotImplementedError

        if frisbee.config.q_matching:
            # TODO: Fix for more than one attention head
            q_matching_loss = frisbee.config.q_matching_loss_coef * F.mse_loss(tilde_q_all_actions.squeeze(), target_q)
        else:
            q_matching_loss = torch.tensor(0.)

    else:
        matching_loss = q_matching_loss = torch.tensor(0.)

    # Total loss
    loss = q_loss + aux_qs_loss + matching_loss + q_matching_loss

    # Wandb logging for the losses
    wandb.log({'loss/q_loss': q_loss.cpu().data.numpy(),
               'loss/aux_qs_loss': aux_qs_loss.cpu().data.numpy(),
               f'loss/{frisbee.config.matching_loss}_matching_loss': matching_loss.cpu().data.numpy(),
               f'loss/q_matching_loss': q_matching_loss.cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done, commit=False)

    # Optimize the model
    frisbee.optimizer.zero_grad()
    loss.backward()
    for param in frisbee.policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    frisbee.optimizer.step()


def train(frisbee):
    # Main learning loop
    for i_episode in range(frisbee.config.num_episodes):
        # --------------------------------------------------------------------
        # Things we do before every episode
        # --------------------------------------------------------------------

        # Update the episodic step trackers
        frisbee.trackers.episode_step.rewards.append([])
        frisbee.trackers.episode_step.predicate_rewards.append([])
        frisbee.trackers.episode_step.actions.append([])
        # Update the episodic trackers
        frisbee.trackers.episode.durations.append(0)
        frisbee.trackers.episode.returns.append(0)
        frisbee.trackers.episode.predicate_returns.append(np.zeros(frisbee.n_predicates))

        # Initialize the environment and state
        state = frisbee.env.reset()

        # Run the episode: take actions in the environment
        for _ in count():
            # Sample an action to perform
            action = select_epsilon_greedy_action(state, frisbee)

            # Take a step: single action in the environment
            next_state, reward, done, info = frisbee.env.step(action.item())

            # Wrap the reward in a tensor
            reward = torch.tensor([reward], device=frisbee.device, dtype=torch.float32)

            # Compute the predicate rewards and wrap in a tensor
            predicate_rewards = torch.tensor([[e.predict(state, action, next_state) for e in frisbee.predicates]],
                                             device=frisbee.device, dtype=torch.float32)

            # Ensure that the next_state is marked None if the episode terminated
            if done:
                next_state = None

            # Store the transition in memory
            frisbee.memory.push(state, action, next_state, reward, predicate_rewards)

            # Move to the next state
            state = next_state

            # --------------------------------------------------------------------
            # Things we do every step
            # --------------------------------------------------------------------

            # Update the training trackers
            frisbee.trackers.training.steps_done += 1
            # Update the step trackers
            frisbee.trackers.step.episode.append(i_episode)
            # Update the episodic step trackers
            frisbee.trackers.episode_step.rewards[-1].append(reward.item())
            frisbee.trackers.episode_step.predicate_rewards[-1].append(predicate_rewards.cpu().numpy().flatten())
            frisbee.trackers.episode_step.actions[-1].append(action.item())
            # Update the episodic trackers
            frisbee.trackers.episode.durations[-1] += 1
            frisbee.trackers.episode.returns[-1] += reward.item()
            frisbee.trackers.episode.predicate_returns[-1] += predicate_rewards.cpu().numpy().flatten()

            # Optimize the model
            if frisbee.trackers.training.steps_done % frisbee.config.update_freq == 0 or done:
                optimize_model(frisbee)

            # Step update of weights and biases
            step_update_training_wandb(frisbee)

            # Update the target network
            update_target_network(frisbee)

            # Checkpoint the model every so often
            checkpoint_model(frisbee)

            # Break if we're done with the episode
            if done:
                break

        # --------------------------------------------------------------------
        # Things we do after every episode
        # --------------------------------------------------------------------

        # Update the training trackers
        frisbee.trackers.training.episodes_done += 1
        frisbee.trackers.training.best_return = max(frisbee.trackers.training.best_return,
                                                    frisbee.trackers.episode.returns[-1])
        # Update the episodic trackers
        pass

        # Episodic update of weights and biases
        episode_update_wandb(frisbee)


def main(args):
    # Create a frisbee -- a SimpleNamespace to toss around that contains general information
    frisbee = create_training_frisbee(args)

    # Set up the environment and add it to the frisbee
    add_sns_to_frisbee(setup_env(frisbee), frisbee)

    # Set up the model and add it to the frisbee
    add_sns_to_frisbee(setup_model(frisbee), frisbee)

    # Set up the training trackers and add them to the frisbee
    add_sns_to_frisbee(setup_training_trackers(frisbee), frisbee)

    # Set up the predicates and add them to the frisbee
    add_sns_to_frisbee(setup_predicates(frisbee), frisbee)

    # Do an initial update to weights and biases
    initial_update_wandb(frisbee)

    # Train
    train(frisbee)


if __name__ == '__main__':
    # Set up the argument parser
    parser = ArgumentParser('Pass in arguments!')
    parser.add_argument('--config', '-c', help='Path to the config .yaml file.', type=str, required=True)
    args = parser.parse_args()

    # Run!
    main(args)
