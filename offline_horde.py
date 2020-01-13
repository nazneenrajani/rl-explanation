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
    wandb.log({'standard/maxQ': torch.max(prediction.main_demon).cpu().data.numpy(),
               'standard/minQ': torch.min(prediction.main_demon).cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done, commit=False)

    for i, g in enumerate(frisbee.config.pred_gammas):
        for j, predicate in enumerate(frisbee.predicates):
            wandb.log({f'prediction/{predicate.name()}@{int(1/(1-g))}_maxQ':
                           torch.max(prediction.prediction_demons[:, i * frisbee.n_predicates + j]).cpu().data.numpy(),
                       f'prediction/{predicate.name()}@{int(1/(1-g))}_minQ':
                           torch.min(prediction.prediction_demons[:, i * frisbee.n_predicates + j]).cpu().data.numpy()},
                      step=frisbee.trackers.training.steps_done, commit=False)
            wandb.log({f'control/{predicate.name()}@{int(1 / (1 - g))}_maxQ':
                           torch.max(prediction.control_demons[:, i * frisbee.n_predicates + j]).cpu().data.numpy(),
                       f'control/{predicate.name()}@{int(1 / (1 - g))}_minQ':
                           torch.min(prediction.control_demons[:, i * frisbee.n_predicates + j]).cpu().data.numpy()},
                      step=frisbee.trackers.training.steps_done, commit=False)


def optimize_auxiliary_model(frisbee):
    # --------------------------------------------------------------------
    # Sample from replay memory
    # --------------------------------------------------------------------
    if len(frisbee.memory) < frisbee.config.batch_size:
        return

    transitions = frisbee.memory.sample(frisbee.config.batch_size,
                                        (frisbee.config.replay_buffer_beta_end -
                                         frisbee.config.replay_buffer_beta_start) /
                                        frisbee.config.replay_buffer_beta_decay
                                        + frisbee.config.replay_buffer_beta_start)

    batch = PrioritizedPredicateTransition(*zip(*transitions))

    # Construct the batches
    state_batch = torch.tensor(batch.state, dtype=torch.float32, device=frisbee.device)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=frisbee.device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=frisbee.device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=frisbee.device)
    predicate_rewards_batch = torch.tensor(batch.predicate_rewards, dtype=torch.float32, device=frisbee.device)
    done_batch = torch.tensor(batch.done, dtype=torch.uint8, device=frisbee.device)
    indices_batch = batch.index
    weights_batch = torch.tensor(batch.weight, dtype=torch.float32, device=frisbee.device).unsqueeze(1)

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
        prediction_demons_at_actions = policy_net_prediction.prediction_demons.gather(2, action_batch.unsqueeze(1).
                                                                                      expand(action_batch.shape[0],
                                                                                             frisbee.n_heads,
                                                                                             action_batch.shape[1]))

        control_demons_at_actions = policy_net_prediction.control_demons.gather(2, action_batch.unsqueeze(1).
                                                                                expand(action_batch.shape[0],
                                                                                       frisbee.n_heads,
                                                                                       action_batch.shape[1]))

    # Wandb logging
    if frisbee.trackers.training.steps_done % 100 == 0:
        prediction_update_wandb(policy_net_prediction, frisbee)

    # --------------------------------------------------------------------
    # Predict using the target network (at non-final next states)
    # --------------------------------------------------------------------

    target_net_prediction = frisbee.target_net(non_final_next_states)
    if frisbee.config.q_target == 'double':
        policy_net_prediction_q = frisbee.policy_net.forward_q(non_final_next_states)

    # Calculate the value function at s'
    v_next = torch.zeros(frisbee.config.batch_size, device=frisbee.device)

    if frisbee.config.q_target == 'standard':
        v_next[non_final_mask] = target_net_prediction.main_demon.max(1)[0].detach()
    elif frisbee.config.q_target == 'double':
        v_next[non_final_mask] = target_net_prediction.main_demon[range(non_final_mask.shape[0]),
                                                                  policy_net_prediction_q.argmax(1)].detach()
    else:
        raise NotImplementedError

    if frisbee.n_predicates > 0:
        # Calculate the auxiliary value functions at s'
        aux_vs_next = torch.zeros((frisbee.config.batch_size, frisbee.n_heads), device=frisbee.device)

        # Evaluation operator with greedy policy from main Q
        if frisbee.config.q_target == 'standard':
            aux_vs_next[non_final_mask] = target_net_prediction.prediction_demons.detach(). \
                gather(dim=-1, index=target_net_prediction.main_demon.argmax(1).detach().unsqueeze(1).
                       expand(target_net_prediction.prediction_demons.shape[:-1]).unsqueeze(2)).squeeze(2)
        elif frisbee.config.q_target == 'double':
            aux_vs_next[non_final_mask] = target_net_prediction.prediction_demons.detach(). \
                gather(dim=-1, index=policy_net_prediction_q.argmax(1).detach().unsqueeze(1).
                       expand(target_net_prediction.prediction_demons.shape[:-1]).unsqueeze(2)).squeeze(2)

        # Off-policy optimality operator
        aux_vs_greedy_next = torch.zeros((frisbee.config.batch_size, frisbee.n_heads), device=frisbee.device)

        if frisbee.config.q_target == 'standard':
            # This is a biased update
            aux_vs_greedy_next[non_final_mask] = target_net_prediction.control_demons.max(2)[0].detach()
        else:
            raise NotImplementedError

    # --------------------------------------------------------------------
    # Compute the losses
    # --------------------------------------------------------------------

    pred_gammas = torch.tensor(frisbee.config.pred_gammas).repeat_interleave(frisbee.n_predicates).to(frisbee.device)
    predicate_rewards_batch_repeated = predicate_rewards_batch.repeat(1, len(frisbee.config.pred_gammas))

    # Compute the targets for the Bellman updates
    target_q_at_actions = (v_next * frisbee.config.gamma) + reward_batch
    if frisbee.n_predicates > 0:
        target_aux_qs_at_actions = (aux_vs_next * pred_gammas) + predicate_rewards_batch_repeated
        target_aux_qs_greedy_at_actions = (aux_vs_greedy_next * pred_gammas) + predicate_rewards_batch_repeated

        # Clamp the values because no Q-values should be negative for binary predicates
        target_aux_qs_at_actions = torch.clamp_min(target_aux_qs_at_actions, 0.)
        target_aux_qs_greedy_at_actions = torch.clamp_min(target_aux_qs_greedy_at_actions, 0.)

        # Clamp the values based on the effective horizon (R_max * effective horizon)
        target_aux_qs_at_actions = torch.min(target_aux_qs_at_actions, 1./(1. - pred_gammas))
        target_aux_qs_greedy_at_actions = torch.min(target_aux_qs_greedy_at_actions, 1./(1. - pred_gammas))

    # Calculate the Q-learning loss
    q_loss = (F.smooth_l1_loss(main_demon_at_actions, target_q_at_actions.unsqueeze(1), reduction='none')
              * weights_batch).mean()

    if frisbee.n_predicates > 0:
        aux_qs_loss = (F.smooth_l1_loss(prediction_demons_at_actions,
                                        target_aux_qs_at_actions.unsqueeze(2), reduction='none')
                       * weights_batch.unsqueeze(1)).mean()

        aux_qs_greedy_loss = (F.smooth_l1_loss(control_demons_at_actions,
                                               target_aux_qs_greedy_at_actions.unsqueeze(2), reduction='none')
                              * weights_batch.unsqueeze(1)).mean()

    else:
        aux_qs_loss = torch.tensor(0.)
        aux_qs_greedy_loss = torch.tensor(0.)

    # Update the replay memory priorities
    abs_td_errors = torch.abs(main_demon_at_actions.squeeze() - target_q_at_actions).cpu().data.numpy() + 1e-2
    frisbee.memory.update_priorities(indices_batch, abs_td_errors)

    # Total loss
    loss = aux_qs_loss + aux_qs_greedy_loss

    # Wandb logging for the losses
    wandb.log({'loss/q_loss': q_loss.cpu().data.numpy(),
               'loss/aux_qs_loss': aux_qs_loss.cpu().data.numpy(),
               'loss/aux_qs_greedy_loss': aux_qs_loss.cpu().data.numpy()},
              step=frisbee.trackers.training.steps_done)

    # Optimize the model
    frisbee.optimizer.zero_grad()
    loss.backward()
    for param in frisbee.policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    frisbee.optimizer.step()


def load_phase_1_run(frisbee):
    # Evaluate
    api = wandb.Api()
    org = frisbee.config.wandb_org
    project = frisbee.config.wandb_project.replace("phase-2", "phase-1")

    # Load up the run
    frisbee.phase_1_run = api.run("/".join([org, project, frisbee.config.phase_1_run_id]))
    print("Picked run:", frisbee.phase_1_run.id)


def load_phase_1_data(frisbee):
    # Load up the phase 1 run
    load_phase_1_run(frisbee)

    # Find all the checkpoint files associated with the phase 1 run
    run_files = frisbee.phase_1_run.files()
    print(run_files)
    print([e.name for e in run_files])
    try:
        pmodel_files = [e for e in run_files if e.name.endswith('.pmodel')]
        pickled_stores = [e for e in run_files if e.name.startswith('experience')]
    except TypeError:
        print("Found no files. Exiting.")
        exit()
    if len(pmodel_files) == 0 or len(pickled_stores) == 0:
        print("Found no files. Exiting.")
        exit()

    # Download all the experience
    for store_file in natsorted(pickled_stores, key=lambda e: e.name.rstrip(".p").split("_")[-1]):
        print(f'Downloaded {store_file.name}')
        store_file.download(replace=True, root=wandb.run.dir)
        if int(store_file.name.rstrip(".p").split("_")[-2]) > frisbee.config.phase_1_checkpoint != -1:
            break

    # Populate the experience
    repopulate_experience(frisbee)

    if frisbee.config.phase_1_checkpoint != -1:
        pmodel_files = [e for e in pmodel_files if
                        int(e.name.rstrip('.pmodel').split("_")[-1]) == frisbee.config.phase_1_checkpoint]
    else:
        pmodel_files = [list(natsorted(pmodel_files, key=lambda e: e.name))[-1]]

    # Go over the models
    for pfile in natsorted(pmodel_files, key=lambda e: e.name):
        pmodel = pfile.download(replace=True, root=wandb.run.dir)
        # Disable strict loading, so that only the overlapping parameters are updated
        frisbee.policy_net.load_state_dict(torch.load(pmodel.name, map_location=frisbee.device), strict=False)
        frisbee.target_net.load_state_dict(torch.load(pmodel.name, map_location=frisbee.device), strict=False)

        file_steps = int(pfile.name.rstrip('.pmodel').split("_")[-1])
        frisbee.checkpoint_at = file_steps


def repopulate_experience(frisbee):
    # Restore the experience
    frisbee.experience.unpickle_partial(store_folder=wandb.run.dir + '/experience')

    # Populate the replay buffer
    states = frisbee.env.reset(seeds=frisbee.experience.get_parallel_seeds(frisbee.config.n_envs))

    start = time.time()
    for step in count(0, frisbee.config.n_envs):
        print(f"Populating step {step}, {time.time() - start}s elapsed.", flush=True) if step % 102400 == 0 else None

        # Get the actions
        actions = np.array(frisbee.experience.get_parallel_actions(frisbee.config.n_envs))

        # Some positions may have no actions
        noops_at = (actions == None)
        actions_at = (actions != None)

        if np.all(noops_at):
            # We're done with adding the experience into the buffer
            break

        # Placeholder actions
        actions[noops_at] = 0

        # Take a step
        next_states, rewards, dones, infos = frisbee.env.step(actions)

        # Extract the predicate rewards
        predicate_rewards = np.array([e['predicates'] for e in infos])

        # Restrict to positions which had valid actions
        next_states_at_actions = np.array(next_states)[actions_at]
        rewards = np.array(rewards)[actions_at]
        dones = np.array(dones)[actions_at]
        actions = actions[actions_at]
        predicate_rewards = predicate_rewards[actions_at]
        states = states[actions_at]

        # Store the transitions in memory
        for e in zip(states, actions, next_states_at_actions, rewards, predicate_rewards, dones):
            frisbee.memory.push(*e)

        states = next_states

        # Check if we're done with any episode
        if np.any(dones):
            # Reset the state
            indices = np.where(dones)[0]
            seeds = frisbee.experience.get_parallel_seeds(frisbee.config.n_envs)
            states[indices] = frisbee.env.reset_subset(indices, seeds=[seeds[i] for i in indices])

        if frisbee.config.phase_1_checkpoint != -1 and step > frisbee.config.phase_1_checkpoint:
            # Populated
            break


def train(frisbee):
    # Load up the data and model learned during phase 1
    load_phase_1_data(frisbee)

    # Copy the convnet learned in phase 1 as a good initialization
    frisbee.policy_net.copy_convnet_to_demon_convnet()
    frisbee.target_net.copy_convnet_to_demon_convnet()

    # Use data parallelism
    # frisbee.policy_net = nn.DataParallel(frisbee.policy_net)
    # frisbee.target_net = nn.DataParallel(frisbee.target_net)
    # frisbee.dist_net = nn.DataParallel(frisbee.dist_net)

    # Main learning loop
    for _ in count(0):

        # Optimize the model
        optimize_auxiliary_model(frisbee)

        # Update the target network
        update_target_network(frisbee)

        # Checkpoint the model every so often
        checkpoint_horde(frisbee)

        # Update the training trackers
        frisbee.trackers.training.steps_done += 1

        # Update wandb
        step_update_training_horde_wandb(frisbee)

        if frisbee.trackers.training.steps_done > frisbee.config.num_steps:
            break

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
