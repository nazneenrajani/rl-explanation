import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils import *
from constants import *
from natsort import natsorted
from general import add_sns_to_frisbee, setup_evaluation_trackers, setup_predicates, \
    setup_horde_eval_env, setup_horde_model
from distill_horde import setup_masks
from itertools import count
sys.path.append('..')

CMAP = plt.get_cmap('YlOrRd')


def visualize_episode_step(frisbee, step):
    # Grab all the relevant variables
    qs = [e.main_demon.cpu().numpy() for e in frisbee.trackers.episode_step.predictions[-1]]
    aux_qs = [e.prediction_demons.cpu().numpy().squeeze()[frisbee.prediction_demons_mask_inverted] for e in frisbee.trackers.episode_step.predictions[-1]]
    aux_cqs = [e.control_demons.cpu().numpy().squeeze()[frisbee.control_demons_mask_inverted] for e in frisbee.trackers.episode_step.predictions[-1]]
    tilde_qs = [e.output.cpu().numpy() for e in frisbee.trackers.episode_step.dist_predictions[-1]]
    atts = [e.weights.detach().cpu().numpy().squeeze(1).transpose(1, 0)[torch.cat([frisbee.prediction_demons_mask_inverted,
                                                                frisbee.n_heads + frisbee.control_demons_mask_inverted])]
            for e in frisbee.trackers.episode_step.dist_predictions[-1]]


    n_atts = atts[0].shape[0]

    states = frisbee.trackers.episode_step.renders[-1]
    actions = frisbee.trackers.episode_step.actions[-1]
    rewards = frisbee.trackers.episode_step.rewards[-1]
    predicate_rewards = frisbee.trackers.episode_step.predicate_rewards[-1]

    q_max, q_min = np.max([np.max(qs), np.max(tilde_qs)]), np.min([np.min(qs), np.min(tilde_qs)])
    aux_qs_max, aux_qs_min = np.max(aux_qs), np.min(aux_qs)
    all_rewards_max, all_rewards_min = np.max([np.max(rewards), np.max(predicate_rewards)]), \
                                       np.min([np.min(rewards), np.min(predicate_rewards)])

    if not frisbee.config.attn_softmax:
        atts_max, atts_min = np.max(atts), np.min(atts)
        print (atts_max, atts_min)
    else:
        atts_max, atts_min = 1., 0.

    pred_labels = [e.name() + f'@{int(1 / (1 - g))}' for g in frisbee.config.pred_gammas for e in frisbee.predicates]

    for i in [step]:

        plt.rc('font', size=14)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(states[i])
        axes[0].axis('off')
        axes[0].title.set_text('Current state')
        axes[0].title.set_size(20)
        axes[1].imshow(states[i + 1])
        axes[1].axis('off')
        axes[1].title.set_text('Next state')
        axes[1].title.set_size(20)
        plt.show()

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(np.round(np.concatenate([[rewards[i]], predicate_rewards[i]])[np.newaxis, :], 3),
                    all_rewards_min, all_rewards_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False, ax=ax)
        ax.set_xticklabels(['Task'] + [e.name() for e in frisbee.predicates], rotation=30, ha='right')
        ax.set_yticklabels(['Reward'], rotation=0)
        ax.title.set_text('Rewards')
        ax.title.set_size(20)
        plt.show()

        fig, axes = plt.subplots(2, 1)
        sns.heatmap(np.round(qs[i].squeeze()[np.newaxis, :], 3), q_min, q_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False, ax=axes[0])
        axes[0].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[0].set_yticklabels(['Main'], rotation=0)
        axes[0].title.set_text('Q function')
        axes[0].title.set_size(20)

        sns.heatmap(np.round(tilde_qs[i].squeeze()[np.newaxis, :], 3), q_min, q_max, CMAP, linewidths=3,
                    linecolor='black', annot=True, square=True, cbar=False, ax=axes[1])
        axes[1].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[1].set_yticklabels([f'Attention Weighted' for e in range(tilde_qs[i].shape[-2])], rotation=0)
        axes[1].title.set_text('Q function')
        axes[1].title.set_size(20)
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(max(aux_qs[i].shape[1] / 2, 20), max(aux_qs[i].shape[1] / 2, 20)),
                                 gridspec_kw={'width_ratios': [1, 1], "wspace": 1})
        sns.heatmap(np.round(aux_qs[i], 3),
                    aux_qs_min, aux_qs_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=False, cbar=False, ax=axes[0])
        axes[0].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[0].set_yticklabels(np.array(pred_labels)[frisbee.prediction_demons_mask_inverted], rotation=0)
        axes[0].title.set_text('Predicate Q functions (Prediction)')
        axes[0].title.set_size(20)

        sns.heatmap(np.round(aux_cqs[i], 3),
                    aux_qs_min, aux_qs_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=False, cbar=False, ax=axes[1])
        axes[1].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[1].set_yticklabels(np.array(pred_labels)[frisbee.control_demons_mask_inverted], rotation=0)
        axes[1].title.set_text('Predicate Q function (Control)')
        axes[1].title.set_size(20)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 40))
        sns.heatmap(np.round(atts[i], 3),
                    atts_min, atts_max, CMAP, linewidths=3,
                    linecolor='black', annot=True, square=False, cbar=False, ax=ax)

        # ax.set_yticklabels([f'Attention Head {e + 1}' for e in range(tilde_qs[i].shape[-2])], rotation=0)
        ax.set_xticklabels(MINIGRID_ACTIONS, rotation=30)
        ax.set_yticklabels([e + 'P' for e in np.array(pred_labels)[frisbee.prediction_demons_mask_inverted]] +
                           [e + 'C' for e in np.array(pred_labels)[frisbee.control_demons_mask_inverted]],
                           rotation=0, ha='right')
        ax.title.set_text('Attention Weights')
        ax.title.set_size(20)
        #         ax.axis('off')
        plt.show()


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

    # Check if we're running on GPU
    frisbee.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all the seeds
    set_seeds(frisbee.config.seed)

    return frisbee


def fetch_and_load_models(frisbee):
    # Evaluate
    api = wandb.Api()
    org = frisbee.config.wandb_org
    project = frisbee.config.wandb_project

    # Load up the training run
    runs = api.runs("/".join([org, project]),
                    {"$and": [{f"config.wandb_tags": f"{frisbee.config.wandb_tags.rstrip('/eval') + '/train'}"}]})
    run = runs[0]
    print("Runs are:", [e.id for e in runs])
    print("Picked run:", run.id)

    # Find all the checkpoint files associated with this
    run_files = run.files()
    try:
        pmodel_files = [e for e in run_files if e.name.endswith('.pmodel') and not e.name.startswith('models/')]
        dmodel_files = [e for e in run_files if e.name.endswith('.dmodel') and not e.name.startswith('models/')]
        print (pmodel_files, dmodel_files)
    except TypeError:
        print("Found no model files.")
        return

    if len(pmodel_files) == 0:
        print("Found no model files.")
        return

    if frisbee.args.checkpoint != -1:
        pmodel_files = [e for e in pmodel_files if int(e.name.rstrip('.pmodel').split("_")[-1]) == frisbee.args.checkpoint]
        dmodel_files = [e for e in dmodel_files if int(e.name.rstrip('.dmodel').split("_")[-1]) == frisbee.args.checkpoint]

    for pfile, dfile in zip(natsorted(pmodel_files, key=lambda e: e.name),
                            natsorted(dmodel_files, key=lambda e: e.name)):
        file_steps = int(pfile.name.rstrip('.pmodel').split("_")[-1])
        assert (file_steps == int(dfile.name.rstrip('.dmodel').split("_")[-1]))
        print(f"\nLoading checkpoint at {file_steps} steps. Clearing old evaluation information.")

        # Set up the evaluation trackers and add them to the frisbee
        add_sns_to_frisbee(setup_evaluation_trackers(frisbee), frisbee)

        # Download the model and restore it
        print(f"\nDownloading model from wandb and loading...")
        os.makedirs('evaluation/data', exist_ok=True)
        pmodel = pfile.download(replace=True, root='evaluation/data')
        dmodel = dfile.download(replace=True, root='evaluation/data')

        print(f"\nLoad policy model from {pfile}...")
        frisbee.policy_net.load_state_dict(torch.load(pmodel.name, map_location=frisbee.device), strict=False)

        print(f"\nLoad distilled model from {dfile}...")
        frisbee.dist_net.load_state_dict(torch.load(dmodel.name, map_location=frisbee.device), strict=False)
        frisbee.checkpoint_at = file_steps

        break

    return frisbee


def evaluate(frisbee):

    setup_masks(frisbee)

    # Main evaluation loop
    for i in range(frisbee.config.eval_eps):
        # Initialize the environment and state
        state = frisbee.env.reset()

        # --------------------------------------------------------------------
        # Things we do before every episode
        # --------------------------------------------------------------------

        # Update the episodic step trackers
        frisbee.trackers.episode_step.rewards.append([])
        frisbee.trackers.episode_step.predicate_rewards.append([])
        frisbee.trackers.episode_step.actions.append([])
        frisbee.trackers.episode_step.states.append([state])
        frisbee.trackers.episode_step.renders.append([frisbee.env.render(mode='rgb_array').squeeze()])
        frisbee.trackers.episode_step.plots.append([])
        frisbee.trackers.episode_step.predictions.append([])
        frisbee.trackers.episode_step.dist_predictions.append([])
        # Update the episodic trackers
        frisbee.trackers.episode.durations.append(0)
        frisbee.trackers.episode.returns.append(0)
        frisbee.trackers.episode.predicate_returns.append(np.zeros(frisbee.n_predicates))

        # Run the episode: take actions in the environment
        for _ in count():

            # Make a prediction using the policy network
            prediction = frisbee.policy_net(torch.tensor(state, device=frisbee.device,
                                                         dtype=torch.float32).unsqueeze(0))

            try:
                # Zero out all the predicates that aren't being used in the prediction
                prediction.prediction_demons[:, frisbee.prediction_demons_mask] = 0.
                prediction.control_demons[:, frisbee.control_demons_mask] = 0.
                demons = torch.cat([prediction.prediction_demons, prediction.control_demons], dim=1)

                # Make a prediction using the distillation network
                dist_prediction = frisbee.dist_net(prediction.demon_embedding,
                                                   prediction.main_demon,
                                                   demons)
            except TypeError:
                dist_prediction = None
                pass

            # Get the greedy action using the main Q function
            if frisbee.args.policy == 'main':
                greedy_action = prediction.main_demon.max(1)[1].item()
            elif frisbee.args.policy == 'dist':
                greedy_action = dist_prediction.output.reshape(prediction.main_demon.shape).max(1)[1].item()
            else:
                raise NotImplementedError

            if np.random.rand() < 1 - frisbee.eval_egreedy_eps:
                action = greedy_action
            else:
                action = np.random.randint(0, frisbee.env.action_space.n)

            # Take a step: single action in the environment
            next_state, reward, done, info = frisbee.env.step(action)

            # Compute the predicate rewards
            predicate_rewards = info['predicates']

            # Move to the next state
            state = next_state

            # --------------------------------------------------------------------
            # Things we do every step
            # --------------------------------------------------------------------

            # Update the evaluation trackers
            frisbee.trackers.evaluation.steps_done += 1
            # Update the episodic step trackers
            frisbee.trackers.episode_step.rewards[-1].append(reward)
            frisbee.trackers.episode_step.predicate_rewards[-1].append(predicate_rewards)
            frisbee.trackers.episode_step.actions[-1].append(greedy_action)
            frisbee.trackers.episode_step.states[-1].append(state)
            frisbee.trackers.episode_step.renders[-1].append(frisbee.env.render(mode='rgb_array').squeeze())
            # frisbee.trackers.episode_step.plots[-1].append(plot)
            frisbee.trackers.episode_step.predictions[-1].append(prediction)
            frisbee.trackers.episode_step.dist_predictions[-1].append(dist_prediction)
            # Update the episodic trackers
            frisbee.trackers.episode.durations[-1] += 1
            frisbee.trackers.episode.returns[-1] += reward
            frisbee.trackers.episode.predicate_returns[-1] += predicate_rewards

            # Break if we're done with the episode
            if done:
                break

        # --------------------------------------------------------------------
        # Things we do after every episode
        # --------------------------------------------------------------------

        # Update the evaluation trackers
        frisbee.trackers.evaluation.episodes_done += 1
        frisbee.trackers.evaluation.best_return = max(frisbee.trackers.evaluation.best_return,
                                                      frisbee.trackers.episode.returns[-1])
        # Update the episodic trackers
        pass

    print("Evaluation Statistics")
    print("-----------------------------------")
    print(f"Evaluation Episodes: {frisbee.config.eval_eps}")
    print(f"Mean Return: {np.mean(frisbee.trackers.episode.returns)}")
    print(f"Mean Durations: {np.mean(frisbee.trackers.episode.durations)}")

    return frisbee


def get_evaluation_frisbee(config_file, checkpoint, policy):
    args = SimpleNamespace(config=config_file, checkpoint=checkpoint, policy=policy, background=False)

    # Create a frisbee -- a SimpleNamespace to toss around that contains general information
    return create_evaluation_frisbee(args)


def initial_setup(frisbee):
    # Set up the predicates and add them to the frisbee
    add_sns_to_frisbee(setup_predicates(frisbee), frisbee)

    # Set up the environment and add it to the frisbee
    add_sns_to_frisbee(setup_horde_eval_env(frisbee), frisbee)

    # Set up the model and add it to the frisbee
    add_sns_to_frisbee(setup_horde_model(frisbee), frisbee)

    # Put the policy network and distillation network into evaluation mode
    frisbee.policy_net = frisbee.policy_net.eval()
    frisbee.dist_net = frisbee.dist_net.eval()
    torch.set_grad_enabled(False)

    return frisbee


def environment_info(frisbee):
    print("Information about Environment")
    print("-----------------------------------")
    print("Environment:", frisbee.config.env_name)
    print("Seeds:", frisbee.config.env_seeds)
    print("Wrappers:")
    for i, (w, arg) in enumerate(zip(frisbee.config.env_wrappers, frisbee.config.env_wrapper_args)):
        print(f'\t{i+1}. {w}({arg})')


def predicate_info(frisbee):
    print("Information about Predicates")
    print("-----------------------------------")
    print("Predicates:")
    for i, (p, arg) in enumerate(zip(frisbee.config.predicates, frisbee.config.predicate_args)):
        print(f'\t{i+1}. {p}({arg})')
