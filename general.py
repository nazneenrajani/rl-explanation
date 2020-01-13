import gym
import wandb
import os
import typing as t

from envs import *
from replay import *
from utils import *
from model import *
from predicates import *
from torch.optim import RMSprop
from gym.wrappers import *
from gym_minigrid.wrappers import *


def select_epsilon_greedy_action(state, frisbee):

    with torch.no_grad():
        q = frisbee.policy_net.forward_q(torch.tensor(np.array(state), dtype=torch.float32, device=frisbee.device))
        action = frisbee.pi.sample(q, frisbee.trackers.training.steps_done).flatten()
        wandb.log({'step/epsilon': frisbee.pi.eps}, step=frisbee.trackers.training.steps_done)
        return action


def create_training_frisbee(args):
    # Create a SimpleNamespace that keeps track of everything and gets passed around
    frisbee = SimpleNamespace()

    # Add in the arguments
    frisbee.args = args

    # Check the path to rl-explanation
    frisbee.rlexp_path = get_path_to_rlexp()

    # Load up the config and the defaults
    frisbee.config = load_config(path=args.config)
    frisbee.config.wandb_tags += '/train'

    try:
        frisbee.template_config = load_config(path=frisbee.rlexp_path +
                                                   f'/config/template_{frisbee.config.config_for}.yaml')
    except AttributeError:
        raise AttributeError('The attribute config_for must be defined in the config file.')

    # Do a diff on the template and the user's config and add the diff to the config
    frisbee.diff = config_diff(frisbee.config, frisbee.template_config)

    # Add the diff to the config
    frisbee.config.__dict__.update(frisbee.diff.__dict__)

    # Set up weights and biases and add it to the frisbee
    add_sns_to_frisbee(initialize_wandb(project=frisbee.config.wandb_project,
                                        tags=frisbee.config.wandb_tags.split("/"),
                                        path=frisbee.rlexp_path),
                       frisbee)

    # Update weights and biases with config information
    wandb.config.update(frisbee.config)
    wandb.log({'Defaults': wandb.Table(data=frisbee.diff.__dict__.items(),
                                       columns=['Parameter', 'Value'])}, step=0)

    # Check if we're running on GPU
    frisbee.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all the seeds
    set_seeds(frisbee.config.seed)

    return frisbee


def create_evaluation_frisbee(args) -> SimpleNamespace:
    # Create a SimpleNamespace that keeps track of everything and gets passed around
    frisbee = SimpleNamespace()

    # Add in the arguments
    frisbee.args = args
    frisbee.background = args.background

    # Check the path to rl-explanation
    frisbee.rlexp_path = get_path_to_rlexp()

    # Load up the config and the defaults
    frisbee.config = load_config(path=args.config)
    frisbee.config.wandb_tags += '/eval'
    try:
        frisbee.template_config = load_config(path=frisbee.rlexp_path +
                                                   f'/config/template_{frisbee.config.config_for}.yaml')
    except AttributeError:
        raise AttributeError('The attribute config_for must be defined in the config file.')

    # Do a diff on the template and the user's config and add the diff to the config
    frisbee.diff = config_diff(frisbee.config, frisbee.template_config)

    # Add the diff to the config
    frisbee.config.__dict__.update(frisbee.diff.__dict__)

    # Set up weights and biases and add it to the frisbee
    add_sns_to_frisbee(initialize_wandb(project=frisbee.config.wandb_project,
                                        tags=frisbee.config.wandb_tags.split("/"),
                                        path=frisbee.rlexp_path),
                       frisbee)

    # Update weights and biases with config information
    wandb.config.update(frisbee.config)
    wandb.log({'Defaults': wandb.Table(data=frisbee.diff.__dict__.items(),
                                       columns=['Parameter', 'Value'])}, step=0)

    # Check if we're running on GPU
    frisbee.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all the seeds
    set_seeds(frisbee.config.seed)

    return frisbee


def add_sns_to_frisbee(sns, frisbee):
    frisbee.__dict__.update(sns.__dict__)


def initial_update_wandb(frisbee):
    # Log a rendered observation into wandb
    frisbee.env.reset()
    wandb.log({'example_obs': [wandb.Image(frisbee.env.render(mode='rgb_array').squeeze())]},
              step=frisbee.trackers.training.steps_done)

    # Set a watch on the policy model to log parameters and gradients
    try:
        wandb.watch((frisbee.policy_net, frisbee.dist_net), log="all")
    except:
        wandb.watch(frisbee.policy_net, log="all")


def watch_model_wandb(frisbee):
    # Set a watch on the policy model to log parameters and gradients
    try:
        wandb.watch((frisbee.policy_net, frisbee.dist_net), log="all")
    except:
        wandb.watch(frisbee.policy_net, log="all")


def episode_update_wandb(frisbee):
    # Log the episode trackers
    wandb.log({f'episode/{e}': frisbee.trackers.episode.__dict__[e][-1]
               for e in frisbee.trackers.episode.__dict__.keys()},
              step=frisbee.trackers.training.steps_done, commit=False)

    # Log the episode step trackers
    wandb.log({f'episode_step/{e}': frisbee.trackers.episode_step.__dict__[e][-1]
               for e in frisbee.trackers.episode_step.__dict__.keys()},
              step=frisbee.trackers.training.steps_done)


def step_update_training_wandb(frisbee):
    # Log the training trackers
    wandb.log({f'training/{e}': frisbee.trackers.training.__dict__[e]
               for e in frisbee.trackers.training.__dict__.keys()},
              step=frisbee.trackers.training.steps_done, commit=False)

    # Log the step trackers
    wandb.log({f'step/{e}': frisbee.trackers.step.__dict__[e][-1]
               for e in frisbee.trackers.step.__dict__.keys()},
              step=frisbee.trackers.training.steps_done)


def step_update_training_horde_wandb(frisbee):
    # Log the training trackers
    print ({f'training/{e}': frisbee.trackers.training.__dict__[e]
               for e in frisbee.trackers.training.__dict__.keys()})
    wandb.log({f'training/{e}': frisbee.trackers.training.__dict__[e]
               for e in frisbee.trackers.training.__dict__.keys()},
              step=frisbee.trackers.training.steps_done, commit=False)


def step_update_evaluation_wandb(frisbee):
    # Log the training trackers
    wandb.log({f'evaluation/{e}': frisbee.trackers.evaluation.__dict__[e]
               for e in frisbee.trackers.evaluation.__dict__.keys()},
              step=frisbee.trackers.evaluation.steps_done, commit=False)

    # Log the step trackers
    wandb.log({f'step/{e}': frisbee.trackers.step.__dict__[e][-1]
               for e in frisbee.trackers.step.__dict__.keys()},
              step=frisbee.trackers.evaluation.steps_done)


'''
Setup functions have a fixed interface

input: a frisbee
output: SimpleNamespace

and populate the SimpleNamespace output with useful information or objects. 
These are (typically) called before any training/evaluation.
'''


def setup_training_trackers(frisbee):
    # Create a SimpleNamespace for keeping track of information generated during training
    trackers = SimpleNamespace()

    # Aggregate information across training
    # --------------------------------------------------------------------
    trackers.training = SimpleNamespace()
    # Number of steps so far: scalar
    trackers.training.steps_done = 0
    # Number of episodes so far: scalar
    trackers.training.episodes_done = 0
    # Best return (in any episode) seen so far: scalar
    trackers.training.best_return = 0

    # Aggregate information from each training episode
    # --------------------------------------------------------------------
    trackers.episode = SimpleNamespace()
    # Duration: scalar
    trackers.episode.durations = []
    # Return at the end: scalar
    trackers.episode.returns = []
    # Predicate returns at the end: vector
    trackers.episode.predicate_returns = []

    # Per-step information from each training episode
    # --------------------------------------------------------------------
    trackers.episode_step = SimpleNamespace()
    # Rewards: list of scalars
    trackers.episode_step.rewards = []
    # Predicate rewards: list of vectors
    trackers.episode_step.predicate_rewards = []
    # Actions: list of scalars
    trackers.episode_step.actions = []

    # Per-step information independent of episode
    # --------------------------------------------------------------------
    trackers.step = SimpleNamespace()
    # Episode: scalar
    trackers.step.episode = []

    return SimpleNamespace(trackers=trackers)


def setup_training_trackers_horde(frisbee):
    # Create a SimpleNamespace for keeping track of information generated during training
    trackers = SimpleNamespace()

    # Aggregate information across training
    # --------------------------------------------------------------------
    trackers.training = SimpleNamespace()
    # Number of steps so far: scalar
    trackers.training.steps_done = 0
    # Number of episodes so far: scalar
    trackers.training.episodes_done = 0
    # Best return (in any episode) seen so far: scalar
    trackers.training.best_return = 0

    # Aggregate information from each training episode
    # --------------------------------------------------------------------
    trackers.episode = SimpleNamespace()
    # Duration: scalar
    trackers.episode.durations = []
    # Return at the end: scalar
    trackers.episode.returns = []
    # Predicate returns at the end: vector
    trackers.episode.predicate_returns = []

    return SimpleNamespace(trackers=trackers)


def setup_evaluation_trackers(frisbee):
    # Create a SimpleNamespace for keeping track of information generated during evaluation
    trackers = SimpleNamespace()

    # Aggregate information across evaluation
    # --------------------------------------------------------------------
    trackers.evaluation = SimpleNamespace()
    # Number of steps so far: scalar
    trackers.evaluation.steps_done = 0
    # Number of episodes so far: scalar
    trackers.evaluation.episodes_done = 0
    # Best return (in any episode) seen so far: scalar
    trackers.evaluation.best_return = 0

    # Aggregate information from each evaluation episode
    # --------------------------------------------------------------------
    trackers.episode = SimpleNamespace()
    # Duration: scalar
    trackers.episode.durations = []
    # Return at the end: scalar
    trackers.episode.returns = []
    # Predicate returns at the end: vector
    trackers.episode.predicate_returns = []

    # Per-step information from each evaluation episode
    # --------------------------------------------------------------------
    trackers.episode_step = SimpleNamespace()
    # Rewards: list of scalars
    trackers.episode_step.rewards = []
    # Predicate rewards: list of vectors
    trackers.episode_step.predicate_rewards = []
    # Actions: list of scalars
    trackers.episode_step.actions = []
    # States: list of tensors
    trackers.episode_step.states = []
    # Renders: list of tensors
    trackers.episode_step.renders = []
    # Plots: list of PIL.Images
    trackers.episode_step.plots = []
    # Predictions: list of namespaces
    trackers.episode_step.predictions = []
    if 'horde' in frisbee.config.config_for:
        # Distillation predictions: list of namespaces
        trackers.episode_step.dist_predictions = []

    return SimpleNamespace(trackers=trackers)


def setup_predicates(frisbee):
    # Set up the list of predicates
    predicates = [globals()[e](*args) for e, args in zip(frisbee.config.predicates, frisbee.config.predicate_args)]
    n_predicates = len(predicates)
    return SimpleNamespace(predicates=predicates, n_predicates=n_predicates)


def setup_env(frisbee):
    # Make the gym environment
    env = gym.make(frisbee.config.env_name)

    # Apply wrappers that were specified
    for wrapper, wrapper_args in zip(frisbee.config.env_wrappers, frisbee.config.env_wrapper_args):
        env = globals()[wrapper](env, *wrapper_args)

    # Wrap stuff in torch.tensors
    env = TorchObsWrapper(env, frisbee.device)

    # Set the seed
    env.seed(frisbee.config.seed)

    return SimpleNamespace(env=env)


def setup_horde_env(frisbee):

    def make_env(seeds, rank):
        # Make the gym environment
        env = gym.make(frisbee.config.env_name)

        # Apply wrappers that were specified
        for wrapper, wrapper_args in zip(frisbee.config.env_wrappers, frisbee.config.env_wrapper_args):
            env = globals()[wrapper](env, *wrapper_args)

        # Reseed the environment
        if frisbee.config.reseed:
            if frisbee.config.reseed_strategy == 'same':
                env = ReseedWrapper(env, seeds=seeds, seed_idx=rank % len(seeds))
            else:
                raise NotImplementedError

        # Predicate wrapper
        env = PredicateWrapper(env, predicates=frisbee.predicates)

        return env

    if frisbee.config.seed_strategy == 'range':
        env = SubprocVecEnv([(lambda x: (lambda: make_env([x], x)))(i) for i in range(frisbee.config.n_envs)])
    elif frisbee.config.seed_strategy == 'same':
        env = SubprocVecEnv([lambda: make_env([1], i) for i in range(frisbee.config.n_envs)])
    elif frisbee.config.seed_strategy == 'specific':
        env = SubprocVecEnv([(lambda x: (lambda: make_env(frisbee.config.env_seeds, x)))(i)
                             for i in range(frisbee.config.n_envs)])
    else:
        raise NotImplementedError

    return SimpleNamespace(env=env)


def setup_horde_eval_env(frisbee):

    def make_env(seeds, rank):
        # Make the gym environment
        env = gym.make(frisbee.config.env_name)

        # Apply wrappers that were specified
        for wrapper, wrapper_args in zip(frisbee.config.env_wrappers, frisbee.config.env_wrapper_args):
            env = globals()[wrapper](env, *wrapper_args)

        # Reseed the environment
        if frisbee.config.reseed:
            if frisbee.config.reseed_strategy == 'same':
                env = ReseedWrapper(env, seeds=seeds, seed_idx=rank % len(seeds))
            else:
                raise NotImplementedError

        # Predicate wrapper
        env = PredicateWrapper(env, predicates=frisbee.predicates)

        return env

    return SimpleNamespace(env=make_env(frisbee.config.env_seeds, 0))


def setup_model(frisbee):
    # Get number of actions from gym action space
    n_actions = frisbee.env.action_space.n

    # Get the shape of the environment's observation
    obs_shape = get_env_observation_shape(frisbee.env)

    # Set up the policy and target networks
    policy_net = globals()[frisbee.config.model](frisbee.device, *obs_shape, len(frisbee.config.predicates),
                                                 n_actions, frisbee.config.n_attention_heads,
                                                 frisbee.config.conv_layers,
                                                 frisbee.config.output_wts,
                                                 frisbee.config.detach_attn_context).to(frisbee.device)

    target_net = globals()[frisbee.config.model](frisbee.device, *obs_shape, len(frisbee.config.predicates),
                                                 n_actions, frisbee.config.n_attention_heads,
                                                 frisbee.config.conv_layers,
                                                 frisbee.config.output_wts,
                                                 frisbee.config.detach_attn_context).to(frisbee.device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Set up the optimizer
    optimizer = globals()[frisbee.config.optimizer](policy_net.parameters(), lr=frisbee.config.lr)

    # Set up the experience replay buffer
    memory = ReplayMemory(frisbee.config.replay_buffer_size)

    # Policy layer for the Q function
    pi = EpsilonGreedyPolicy(frisbee.config.eps_start,
                             frisbee.config.eps_end,
                             frisbee.config.eps_decay).to(frisbee.device)

    # Policy layer for the tilde Q function
    tilde_pi = BoltzmannPolicy(frisbee.config.temp_start,
                               frisbee.config.temp_end,
                               frisbee.config.temp_decay).to(frisbee.device)

    return SimpleNamespace(n_actions=n_actions,
                           obs_shape=obs_shape,
                           policy_net=policy_net,
                           target_net=target_net,
                           optimizer=optimizer,
                           memory=memory,
                           pi=pi,
                           tilde_pi=tilde_pi)


def setup_horde_model(frisbee):
    # Get number of actions from gym action space
    n_actions = frisbee.env.action_space.n

    # Get the shape of the environment's observation
    obs_shape = get_env_observation_shape(frisbee.env)

    # Compute the effective number of predicate heads
    frisbee.n_heads = n_heads = len(frisbee.config.predicates) * len(frisbee.config.pred_gammas)

    # Set up the policy and target networks
    policy_net = globals()[frisbee.config.model](frisbee.device, *obs_shape, n_heads,
                                                 n_actions, frisbee.config.conv_layers,
                                                 frisbee.config.batch_norm,
                                                 frisbee.config.detach_aux_demons,
                                                 frisbee.config.two_streams).to(frisbee.device)

    target_net = globals()[frisbee.config.model](frisbee.device, *obs_shape, n_heads,
                                                 n_actions, frisbee.config.conv_layers,
                                                 frisbee.config.batch_norm,
                                                 frisbee.config.detach_aux_demons,
                                                 frisbee.config.two_streams).to(frisbee.device)

    # Set up the distillation network
    dist_net = globals()[frisbee.config.dist_model](frisbee.device, frisbee.config.k_dim,
                                                    policy_net.linear_input_size, 2 * n_heads,
                                                    n_actions, frisbee.config.attn_mechanism,
                                                    frisbee.config.attn_softmax, frisbee.config.affine_vals,
                                                    frisbee.config.fit_residuals,
                                                    frisbee.config.standardize).to(frisbee.device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Set up the optimizers
    optimizer = globals()[frisbee.config.optimizer](policy_net.parameters(), lr=frisbee.config.lr)
    dist_optimizer = globals()[frisbee.config.dist_optimizer](dist_net.parameters(), lr=frisbee.config.dist_lr)

    # Set up the experience replay buffer
    memory = PrioritizedReplayMemory(capacity=frisbee.config.replay_buffer_size,
                                     alpha=frisbee.config.replay_buffer_alpha)

    # Set up the experience store
    experience = ExplorationExperience()

    # Policy layer for the Q function
    pi = EpsilonGreedyPolicy(frisbee.config.eps_start,
                             frisbee.config.eps_end,
                             frisbee.config.eps_decay).to(frisbee.device)

    # Policy layer for the tilde Q function
    tilde_pi = BoltzmannPolicy(frisbee.config.temp_start,
                               frisbee.config.temp_end,
                               frisbee.config.temp_decay).to(frisbee.device)

    return SimpleNamespace(n_actions=n_actions,
                           obs_shape=obs_shape,
                           policy_net=policy_net,
                           target_net=target_net,
                           dist_net=dist_net,
                           optimizer=optimizer,
                           dist_optimizer=dist_optimizer,
                           memory=memory,
                           experience=experience,
                           pi=pi,
                           tilde_pi=tilde_pi)


def update_target_network(frisbee):
    if frisbee.trackers.training.steps_done % frisbee.config.target_update == 0:
        print(f"Updating target network at {frisbee.trackers.training.steps_done} steps")
        frisbee.target_net.load_state_dict(frisbee.policy_net.state_dict())


def checkpoint_model(frisbee):
    if frisbee.trackers.training.steps_done % frisbee.config.checkpoint_freq == 1:
        store_path = os.path.join(wandb.run.dir,
                                  f'models/{frisbee.wandb.id}_{frisbee.trackers.training.steps_done}.model')
        os.makedirs(wandb.run.dir + '/models/', exist_ok=True)
        torch.save(frisbee.policy_net.state_dict(), store_path)
        wandb.save(store_path)


def checkpoint_horde(frisbee, force=False):
    if frisbee.trackers.training.steps_done % frisbee.config.checkpoint_freq == 0 or force:
        # Store the policy model
        os.makedirs(wandb.run.dir + '/models/', exist_ok=True)
        store_path = os.path.join(wandb.run.dir,
                                  f'models/{frisbee.wandb.id}_{frisbee.trackers.training.steps_done}.pmodel')
        if 'parallel' in str(type(frisbee.policy_net)).lower():
            torch.save(frisbee.policy_net.module.state_dict(), store_path)
        else:
            torch.save(frisbee.policy_net.state_dict(), store_path)
        wandb.save(store_path)

        # Store the distilled model
        store_path = os.path.join(wandb.run.dir,
                                  f'models/{frisbee.wandb.id}_{frisbee.trackers.training.steps_done}.dmodel')
        if 'parallel' in str(type(frisbee.dist_net)).lower():
            torch.save(frisbee.dist_net.module.state_dict(), store_path)
        else:
            torch.save(frisbee.dist_net.state_dict(), store_path)
        wandb.save(store_path)

        if frisbee.config.store_experience:
            # Store the experience
            os.makedirs(wandb.run.dir + '/experience/', exist_ok=True)
            store_path = frisbee.experience.pickle_partial(store_folder=wandb.run.dir + '/experience/',
                                                           file_prefix=f'{frisbee.wandb.id}_'
                                                           f'{frisbee.trackers.training.steps_done}')
            wandb.save(store_path)


