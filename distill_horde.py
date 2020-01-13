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
from sklearn.preprocessing import StandardScaler

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


def setup_masks(frisbee):
    if 'predicate_basis_prediction_demons' in frisbee.config.__dict__:
        prediction_demons_mask = torch.ones(frisbee.n_heads).reshape(len(frisbee.config.pred_gammas),
                                                                     frisbee.n_predicates)
        if len(frisbee.config.predicate_basis_prediction_demons) > 0:
            prediction_demons_mask[frisbee.config.pred_gammas_basis_prediction_demon] = \
                prediction_demons_mask[frisbee.config.pred_gammas_basis_prediction_demon]. \
                    scatter(1, torch.tensor(frisbee.config.predicate_basis_prediction_demons).
                            repeat(len(frisbee.config.pred_gammas_basis_prediction_demon), 1), 0)
    else:
        prediction_demons_mask = torch.zeros(frisbee.n_heads).reshape(len(frisbee.config.pred_gammas),
                                                                      frisbee.n_predicates)

    frisbee.prediction_demons_mask = prediction_demons_mask.flatten().nonzero().squeeze().to(frisbee.device)

    if 'predicate_basis_control_demons' in frisbee.config.__dict__:
        control_demons_mask = torch.ones(frisbee.n_heads).reshape(len(frisbee.config.pred_gammas),
                                                                  frisbee.n_predicates)
        if len(frisbee.config.predicate_basis_control_demons) > 0:
            control_demons_mask[frisbee.config.pred_gammas_basis_control_demon] = \
                control_demons_mask[frisbee.config.pred_gammas_basis_control_demon]. \
                    scatter(1, torch.tensor(frisbee.config.predicate_basis_control_demons).
                            repeat(len(frisbee.config.pred_gammas_basis_control_demon), 1), 0)

    else:
        control_demons_mask = torch.zeros(frisbee.n_heads).reshape(len(frisbee.config.pred_gammas),
                                                                   frisbee.n_predicates)

    frisbee.control_demons_mask = control_demons_mask.flatten().nonzero().squeeze().to(frisbee.device)

    frisbee.control_demons_mask_inverted = torch.ones(frisbee.n_heads)
    frisbee.control_demons_mask_inverted[frisbee.control_demons_mask] = 0.
    frisbee.control_demons_mask_inverted = frisbee.control_demons_mask_inverted.nonzero().flatten()

    frisbee.prediction_demons_mask_inverted = torch.ones(frisbee.n_heads)
    frisbee.prediction_demons_mask_inverted[frisbee.prediction_demons_mask] = 0.
    frisbee.prediction_demons_mask_inverted = frisbee.prediction_demons_mask_inverted.nonzero().flatten()


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
        context = prediction.demon_embedding.detach()
    else:
        prediction = frisbee.policy_net(state_batch)
        context = prediction.demon_embedding.detach() if frisbee.config.detach_attn_context else \
            prediction.demon_embedding

    # Zero out all the predicates that aren't being used in the prediction
    prediction.prediction_demons[:, frisbee.prediction_demons_mask] = 0.
    prediction.control_demons[:, frisbee.control_demons_mask] = 0.
    demons = torch.cat([prediction.prediction_demons.detach(), prediction.control_demons.detach()], dim=1)

    # Distillation as regularized regression
    dist_prediction = frisbee.dist_net(context,
                                       prediction.main_demon.detach(),
                                       demons)

    wandb.log({'attention/outputs': wandb.Histogram(dist_prediction.output.cpu().data.numpy().flatten()),
               'attention/targets': wandb.Histogram(prediction.main_demon.cpu().data.numpy().flatten())},
              step=frisbee.trackers.training.steps_done, commit=False)

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

    if frisbee.config.softmax_loss_fn == 'none':
        policy_loss = torch.tensor(0.)
    elif frisbee.config.softmax_loss_fn == 'kl':
        dist_policy = frisbee.tilde_pi.forward(dist_prediction.output.reshape(prediction.main_demon.shape),
                                               frisbee.trackers.training.steps_done)
        main_policy = frisbee.tilde_pi.forward(prediction.main_demon,
                                               frisbee.trackers.training.steps_done)
        policy_loss = frisbee.config.softmax_loss_coef * F.kl_div(dist_policy.log(), main_policy)
    else:
        raise NotImplementedError

    if frisbee.config.regularization == 'l1':
        regularization = frisbee.config.regularization_coef * dist_prediction.weights.abs().sum(1).mean()
    elif frisbee.config.regularization == 'entropy':
        if not frisbee.config.attn_softmax:
            raise AssertionError('Entropy regularization only works with softmax attention weights.')
        regularization = frisbee.config.regularization_coef * (dist_prediction.weights *
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
               'loss/regularization': regularization.cpu().data.numpy() * frisbee.config.regularization_coef,
               'loss/residual_loss': residual_loss.cpu().data.numpy(),
               'loss/total_regression_loss': loss.cpu().data.numpy(),
               'loss/policy_loss': policy_loss.cpu().data.numpy(),
               'attention/weights': wandb.Histogram(dist_prediction.weights.cpu().data.numpy().flatten())},
              step=frisbee.trackers.training.steps_done, commit=False)

    loss = loss + residual_loss + policy_loss

    # Optimize the model
    frisbee.dist_optimizer.zero_grad()
    loss.backward()
    for param in frisbee.dist_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    frisbee.dist_optimizer.step()


def load_phase_1_and_2_runs(frisbee):
    # Evaluate
    api = wandb.Api()
    org = frisbee.config.wandb_org
    phase_1_project = frisbee.config.wandb_project.replace("phase-3", "phase-1")
    phase_2_project = frisbee.config.wandb_project.replace("phase-3", "phase-2")

    # Load up the runs
    frisbee.phase_1_run = api.run("/".join([org, phase_1_project, frisbee.config.phase_1_run_id]))
    print("Picked run (phase 1):", frisbee.phase_1_run.id)

    frisbee.phase_2_run = api.run("/".join([org, phase_2_project, frisbee.config.phase_2_run_id]))
    print("Picked run (phase 2):", frisbee.phase_2_run.id)


def load_phase_1_and_2_data(frisbee):
    # Load up the phase 1 and 2 runs
    load_phase_1_and_2_runs(frisbee)

    # Find all the checkpoint files associated with the phase 1 run
    phase_1_run_files = frisbee.phase_1_run.files()
    phase_2_run_files = frisbee.phase_2_run.files()
    print("Phase 1:", phase_1_run_files)
    print("Phase 2:", phase_2_run_files)
    try:
        # Get the experience from phase 1
        pickled_stores = [e for e in phase_1_run_files if e.name.startswith('experience')]
        # Get the learned models from phase 2
        pmodel_files = [e for e in phase_2_run_files if e.name.endswith('.pmodel') and
                        e.name.startswith(frisbee.phase_2_run.id)]
    except TypeError:
        print("Found no files. Exiting.")
        exit()
    if len(pmodel_files) == 0 or len(pickled_stores) == 0:
        print("Found no files. Exiting.")
        exit()

    # Download all the experience from phase 1
    for store_file in natsorted(pickled_stores, key=lambda e: e.name.rstrip(".p").split("_")[-1]):
        print(f'Downloaded {store_file.name}')
        store_file.download(replace=True, root=wandb.run.dir)
        if int(store_file.name.rstrip(".p").split("_")[-2]) > frisbee.config.phase_1_checkpoint != -1:
            break

    # Populate the experience
    repopulate_experience(frisbee)

    # Download the pre-trained models from phase 2
    if frisbee.config.phase_2_checkpoint != -1:
        pmodel_files = [e for e in pmodel_files if
                        int(e.name.rstrip('.pmodel').split("_")[-1]) == frisbee.config.phase_2_checkpoint]
    else:
        pmodel_files = [list(natsorted(pmodel_files, key=lambda e: e.name))[-1]]
    print(pmodel_files)

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


def estimate_standardization(frisbee):
    memory_size = len(frisbee.memory.memory)

    prediction_demon_scaler = StandardScaler()
    control_demon_scaler = StandardScaler()
    main_demon_scaler = StandardScaler()

    batch_size = 1024

    for i in range(0, memory_size, batch_size):

        indices = list(range(i, min(i + batch_size, memory_size)))

        transitions = frisbee.memory.get_transitions(indices, 1.)
        batch = PrioritizedPredicateTransition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=frisbee.device)

        prediction = frisbee.policy_net(state_batch)

        prediction_demons = prediction.prediction_demons.cpu().data.numpy().reshape(len(indices), -1)
        control_demons = prediction.control_demons.cpu().data.numpy().reshape(len(indices), -1)
        main_demon = prediction.main_demon.cpu().data.numpy().reshape(-1, 1)

        prediction_demon_scaler.partial_fit(prediction_demons)
        control_demon_scaler.partial_fit(control_demons)
        main_demon_scaler.partial_fit(main_demon)

    pmu = prediction_demon_scaler.mean_.reshape(frisbee.n_heads, frisbee.n_actions)
    pstd = prediction_demon_scaler.scale_.reshape(frisbee.n_heads, frisbee.n_actions)

    cmu = control_demon_scaler.mean_.reshape(frisbee.n_heads, frisbee.n_actions)
    cstd = control_demon_scaler.scale_.reshape(frisbee.n_heads, frisbee.n_actions)

    mmu = main_demon_scaler.mean_
    mstd = main_demon_scaler.scale_

    pstd[pstd < 1e-1] = 1.
    cstd[cstd < 1e-1] = 1.
    mstd[mstd < 1e-2] = 1.

    combined_masks = torch.cat([frisbee.prediction_demons_mask_inverted,
                                frisbee.n_heads + frisbee.control_demons_mask_inverted])

    frisbee.dist_net.demon_scale_mean.data[combined_masks] = \
        torch.tensor(np.concatenate([pmu, cmu], axis=0)[combined_masks], dtype=torch.float32, device=frisbee.device)
    frisbee.dist_net.demon_scale_std.data[combined_masks] = \
        torch.tensor(np.concatenate([pstd, cstd], axis=0)[combined_masks], dtype=torch.float32, device=frisbee.device)

    frisbee.dist_net.target_scale_mean.data = torch.tensor(mmu, dtype=torch.float32, device=frisbee.device)
    frisbee.dist_net.target_scale_std.data = torch.tensor(mstd, dtype=torch.float32, device=frisbee.device)


def train(frisbee):

    # Load up the data and model learned during phase 1 and 2
    load_phase_1_and_2_data(frisbee)

    setup_masks(frisbee)
    estimate_standardization(frisbee)

    # Main learning loop
    for _ in count(0):

        # Optimize the model
        optimize_dist_model(frisbee)

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
