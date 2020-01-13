import pickle
from argparse import ArgumentParser
from types import SimpleNamespace
from itertools import count
from general import *
from constants import *
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
import time

#TODO:
# - viz policies
# - viz action taken
# - viz next state

CMAP = plt.get_cmap('YlOrRd')
np.set_printoptions(precision=2, suppress=True)


def visualize_episode(frisbee):
    # Grab all the relevant variables
    qs = [e.q.cpu().numpy() for e in frisbee.trackers.episode_step.predictions[-1]]
    aux_qs = [e.aux_qs.cpu().numpy() for e in frisbee.trackers.episode_step.predictions[-1]]
    tilde_qs = [e.tilde_q.cpu().numpy() for e in frisbee.trackers.episode_step.predictions[-1]]
    atts = [e.attention.cpu().numpy() for e in frisbee.trackers.episode_step.predictions[-1]]
    tilde_pis = [frisbee.tilde_pi(e.tilde_q, frisbee.checkpoint_at).cpu().numpy()
                 for e in frisbee.trackers.episode_step.predictions[-1]]

    states = frisbee.trackers.episode_step.renders[-1]
    actions = frisbee.trackers.episode_step.actions[-1]
    rewards = frisbee.trackers.episode_step.rewards[-1]
    predicate_rewards = frisbee.trackers.episode_step.predicate_rewards[-1]

    q_max, q_min = np.max(qs), np.min(qs)
    aux_qs_max, aux_qs_min = np.max(aux_qs), np.min(aux_qs)
    tilde_qs_max, tilde_qs_min = np.max(tilde_qs), np.min(aux_qs)
    all_rewards_max, all_rewards_min = np.max([np.max(rewards), np.max(predicate_rewards)]), \
                                       np.min([np.min(rewards), np.min(predicate_rewards)])

    images = []
    for i in tqdm(range(frisbee.trackers.episode.durations[-1])):

        plt.rc('font', size=14)
        fig, axes = plt.subplots(8, 1, gridspec_kw={'height_ratios': [8, 1, 1, frisbee.n_predicates,
                                                                      2, 1, 2, 1],
                                                    "hspace": 0.4},
                                 figsize=(17, 34))

        axes[0].imshow(np.concatenate([states[i], states[i+1]], axis=1))
        axes[0].axis('off')
        axes[0].title.set_text('Current State | Next State')
        axes[0].title.set_size(20)

        sns.heatmap(np.round(qs[i].squeeze()[np.newaxis, :], 2), q_min, q_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False, ax=axes[1])
        axes[1].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[1].set_yticklabels(['Main'], rotation=0)
        axes[1].title.set_text('Q function')
        axes[1].title.set_size(20)

        sns.heatmap(np.round(tilde_qs[i].squeeze()[np.newaxis, :], 2), tilde_qs_min, tilde_qs_max, CMAP, linewidths=3,
                    linecolor='black',
                    annot=True, square=True, cbar=False, ax=axes[2])
        axes[2].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[2].set_yticklabels([f'Attention Weighted' for e in range(tilde_qs[i].shape[-2])], rotation=0)
        axes[2].title.set_text('Q function')
        axes[2].title.set_size(20)

        sns.heatmap(np.round(aux_qs[i].squeeze(), 2), aux_qs_min, aux_qs_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False,
                    yticklabels=[e.name() for e in frisbee.predicates], ax=axes[3])
        axes[3].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[3].title.set_text('Q function')
        axes[3].title.set_size(20)

        sns.heatmap(np.round(np.concatenate([tilde_pis[i].squeeze()[np.newaxis, :],
                                    torch.zeros(1, tilde_pis[i].shape[-1]).
                                   scatter_(dim=1, index=torch.tensor(actions[i]).reshape(1, 1), value=1).numpy()],
                                   axis=0), 2),
                    0, 1, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False,
                    yticklabels=['Explainable Policy', 'Greedy Policy'], ax=axes[4])
        axes[4].set_xticklabels(MINIGRID_ACTIONS, rotation=30, ha='right')
        axes[4].title.set_text('Policies')
        axes[4].title.set_size(20)

        sns.heatmap(np.round(atts[i].squeeze()[np.newaxis, :], 2), 0, 1, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False, ax=axes[5])
        axes[5].set_xticklabels([e.name() for e in frisbee.predicates], rotation=30, ha='right')
        axes[5].set_yticklabels([f'Attention Head {e + 1}' for e in range(tilde_qs[i].shape[-2])], rotation=0)
        axes[5].title.set_text('Attention Weights')
        axes[5].title.set_size(20)

        axes[6].axis('off')

        sns.heatmap(np.round(np.concatenate([[rewards[i]], predicate_rewards[i]])[np.newaxis, :], 2),
                    all_rewards_min, all_rewards_max, CMAP, linewidths=3, linecolor='black',
                    annot=True, square=True, cbar=False,
                    xticklabels=['Task'] + [e.name() for e in frisbee.predicates],
                    yticklabels=['Reward'], ax=axes[7])
        axes[7].set_xticklabels(['Task'] + [e.name() for e in frisbee.predicates], rotation=30, ha='right')
        axes[7].set_yticklabels(['Reward'], rotation=0)
        axes[7].title.set_text('Rewards')
        axes[7].title.set_size(20)

        plt.tight_layout()
        plt.savefig(os.path.join(wandb.run.dir,
                                 f'{frisbee.checkpoint_at}_{frisbee.trackers.evaluation.episodes_done + 1}_{i + 1}'))
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        plt.close(fig)
        images.append(image)

    # Render the frames as a gif and store them in wandb
    images[0].save(os.path.join(wandb.run.dir,
                                f'{frisbee.checkpoint_at}_{frisbee.trackers.evaluation.episodes_done + 1}.gif'),
                   format='GIF', append_images=images[1:], save_all=True, duration=500, loop=0)


def plot_prediction(prediction, frisbee):

    def pil_q():
        mesh = torch.cat([prediction.q.squeeze(0).unsqueeze(0),
                          prediction.aux_qs.squeeze(0),
                          prediction.tilde_q.squeeze(0)], dim=0).cpu().numpy()

        mesh = np.repeat(np.repeat(mesh, 400/mesh.shape[1], axis=1), 400/mesh.shape[0], axis=0)

        im = Image.fromarray(np.array(cm.PiYG(mesh/np.max(mesh))[:, :, :3] * 255).astype(np.uint8))
        return im

    def pil_att():
        mesh = prediction.attention.squeeze(0).cpu().numpy()
        im = Image.fromarray(np.array(cm.PiYG(mesh)[:, :, :3] * 255).astype(np.uint8))
        return im.resize((400, 400))

    def pil_state():
        im = Image.fromarray(frisbee.env.render(mode='rgb_array').squeeze())
        return im.resize((400, 400))

    def plot_q():
        fig, axes = plt.subplots(1, 1)

        mesh = torch.cat([prediction.q.squeeze(0).unsqueeze(0),
                          prediction.aux_qs.squeeze(0),
                          prediction.tilde_q.squeeze(0)], dim=0).cpu().numpy()
        im = axes.pcolor(mesh, cmap=plt.get_cmap('YlOrRd'), edgecolors='k', linewidths=3)
        axes.set_xticks(np.arange(prediction.q.shape[-1]) + 0.5)
        axes.set_xticklabels(MINIGRID_ACTIONS, rotation=45)
        axes.set_yticks(np.arange(1 + prediction.aux_qs.shape[-2] + prediction.tilde_q.shape[-2]) + 0.5)
        axes.set_yticklabels(['Vanilla'] + [e.name() for e in frisbee.predicates] +
                             [f'Attention Weighted' for e in range(prediction.tilde_q.shape[-2])])
        fig.colorbar(im, ax=axes, orientation='horizontal')

        fig.tight_layout()
        plt.savefig(os.path.join(wandb.run.dir,
                                 f'{frisbee.checkpoint_at}_{frisbee.trackers.evaluation.episodes_done+1}_'
                                 f'{frisbee.trackers.episode.durations[-1]+1}_q'))

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        im_q = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        plt.close(fig)

        return im_q#.resize((400, 800))

    def plot_att():
        fig, axes = plt.subplots(1, 1)
        mesh = prediction.attention.squeeze(0).cpu().numpy()
        bounds = np.linspace(0., 1., 11)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        im = axes.pcolor(mesh.T, cmap=plt.get_cmap('YlOrRd'), norm=norm, edgecolors='k', linewidths=4, vmin=0., vmax=1.)
        axes.set_yticks(np.arange(prediction.aux_qs.shape[-2]) + 0.5)
        axes.set_yticklabels([e.name() for e in frisbee.predicates])
        axes.set_xticks(np.arange(prediction.attention.shape[-2]) + 0.5)
        axes.set_xticklabels([f'Attention Head {e+1}' for e in range(prediction.attention.shape[-2])])
        fig.colorbar(im, ax=axes, orientation='horizontal')

        fig.tight_layout()
        plt.savefig(os.path.join(wandb.run.dir,
                                 f'{frisbee.checkpoint_at}_{frisbee.trackers.evaluation.episodes_done+1}_'
                                 f'{frisbee.trackers.episode.durations[-1]+1}_att'))

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        im_att = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

        plt.close(fig)

        return im_att#.resize((400, 800))

    imgs = [pil_state(), plot_q(), plot_att()]
    print (imgs)
    h = min([e.size[1] for e in imgs])
    print (h)
    rimgs = [e.resize((int(e.size[0] * h / e.size[1]), h)) for e in imgs]
    print (rimgs)
    imgs_comb = np.hstack((np.asarray(e) for e in rimgs))
    print (imgs_comb)

    # min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    # imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))


    return Image.fromarray(imgs_comb)


def table_prediction(prediction, frisbee):
    table = wandb.Table(columns=['Q Function'] + MINIGRID_ACTIONS)
    col_1 = np.array([['Vanilla'] + frisbee.config.predicates +
                      [f'Attention Weighted' for e in range(prediction.tilde_q.shape[-2])]]).reshape((-1, 1))
    mesh = torch.cat([prediction.q.squeeze(0).unsqueeze(0), prediction.aux_qs.squeeze(0),
                      prediction.tilde_q.squeeze(0)], dim=0).cpu().numpy()
    rows = np.concatenate([col_1, mesh], axis=1)
    table.add_data(rows)
    wandb.log({'tables/Q functions': table}, step=frisbee.checkpoint_at, commit=False)

    table = wandb.Table(columns=MINIGRID_ACTIONS)
    mesh = prediction.attention.squeeze(0).cpu().numpy()
    table.add_data(mesh)
    wandb.log({'tables/Attention': table}, step=frisbee.checkpoint_at, commit=False)


def evaluate(frisbee):

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
        # Update the episodic trackers
        frisbee.trackers.episode.durations.append(0)
        frisbee.trackers.episode.returns.append(0)
        frisbee.trackers.episode.predicate_returns.append(np.zeros(frisbee.n_predicates))

        # Run the episode: take actions in the environment
        for _ in count():

            # Make a prediction using the policy network
            prediction = frisbee.policy_net(state)
            # plot = plot_prediction(prediction, frisbee)

            # Get the greedy action using the main Q function
            greedy_action = prediction.q.max(1)[1].item()

            # Take a step: single action in the environment
            next_state, reward, done, info = frisbee.env.step(greedy_action)

            # Compute the predicate rewards
            predicate_rewards = np.array([e.predict(state, greedy_action, next_state) for e in frisbee.predicates])

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

        if frisbee.visualize:
            visualize_episode(frisbee)
            wandb.save(os.path.join(wandb.run.dir, f'{frisbee.checkpoint_at}_*.gif'))

    # Get the gif frames
    # frames = [[e for e in ep_plots] for ep_plots in frisbee.trackers.episode_step.plots]

    # Render the frames as a gif and store them in wandb
    # [e[0].save(os.path.join(wandb.run.dir, f'{frisbee.checkpoint_at}_{i + 1}.gif'), format='GIF',
    #            append_images=e[1:], save_all=True, duration=200, loop=0) for i, e in enumerate(frames)]
    # wandb.save(os.path.join(wandb.run.dir, f'{frisbee.checkpoint_at}_*.gif'))

    # Store all the trackers
    # Too big to save
    # pickle.dump(frisbee.trackers, open(os.path.join(wandb.run.dir, f'{frisbee.checkpoint_at}.p'), 'wb'))
    # wandb.save(os.path.join(wandb.run.dir, f'{frisbee.checkpoint_at}.p'))

    # Do wandb logging
    wandb.log({'eval/mean_return': np.mean(frisbee.trackers.episode.returns),
               'eval/std_return': np.std(frisbee.trackers.episode.returns),
               'eval/mean_duration': np.mean(frisbee.trackers.episode.durations),
               'eval/std_duration': np.std(frisbee.trackers.episode.durations)},
              step=frisbee.checkpoint_at, commit=False)

    wandb.log({f'eval/mean_return_{frisbee.config.predicates[i]}': e
               for i, e in enumerate(np.mean(np.array(frisbee.trackers.episode.predicate_returns), axis=0))},
              step=frisbee.checkpoint_at, commit=False)

    wandb.log({f'eval/std_return_{frisbee.config.predicates[i]}': e
               for i, e in enumerate(np.std(np.array(frisbee.trackers.episode.predicate_returns), axis=0))},
              step=frisbee.checkpoint_at)

    return np.mean(frisbee.trackers.episode.returns)


def background_evaluation(frisbee):
    # Evaluate
    api = wandb.Api()
    org = frisbee.config.wandb_org
    project = frisbee.config.wandb_project

    best_metric = -np.inf

    # Load up the training run
    runs = api.runs("/".join([org, project]),
                    {"$and": [{f"config.wandb_tags": f"{frisbee.config.wandb_tags.rstrip('/eval') + '/train'}"}]})
    run = runs[0]
    print("Runs are:", [e.id for e in runs])
    print("Picked run:", run.id)

    checkpoints_processed = set()

    # Runs in the background
    while True:
        # Find all the checkpoint files associated with this
        run_files = run.files()
        try:
            model_files = [e for e in run_files if e.name.endswith('.model') and e not in checkpoints_processed]
        except TypeError:
            continue
        if len(model_files) == 0:
            continue

        file = model_files[np.argmin([int(e.name.rstrip('.model').split("_")[-1]) for e in model_files])]
        file_steps = int(file.name.rstrip('.model').split("_")[-1])

        # most_recent_file = model_files[np.argmax([int(e.name.rstrip('.model').split("_")[-1]) for e in model_files])]
        # most_recent_file_steps = int(most_recent_file.name.rstrip('.model').split("_")[-1])

        print(f"Evaluating checkpoint at {file_steps} steps")

        # Set up the evaluation trackers and add them to the frisbee
        add_sns_to_frisbee(setup_evaluation_trackers(frisbee), frisbee)

        # Download the model and restore it
        model = file.download(replace=True, root=wandb.run.dir)
        frisbee.policy_net.load_state_dict(torch.load(model.name, map_location=frisbee.device))
        frisbee.checkpoint_at = file_steps

        # Do evaluation
        metric = evaluate(frisbee)

        # Keep track of the fact that we've used up this model
        checkpoints_processed.add(file)

        if metric > best_metric:
            best_model = file_steps
            best_metric = metric
            wandb.log({'eval/best_model': best_model,
                       'eval/best_return': best_metric},
                      step=frisbee.checkpoint_at)

        # Break out if the training process terminated and we're running this evaluation in the background
        if not run.state == 'running':
            break


def post_evaluation(frisbee):
    # Evaluate
    api = wandb.Api()
    org = frisbee.config.wandb_org
    project = frisbee.config.wandb_project

    best_metric = -np.inf

    # Load up the training run
    runs = api.runs("/".join([org, project]),
                    {"$and": [{f"config.wandb_tags": f"{frisbee.config.wandb_tags.rstrip('/eval') + '/train'}"}]})
    run = runs[0]
    print("Runs are:", [e.id for e in runs])
    print("Picked run:", run.id)

    # Find all the checkpoint files associated with this
    run_files = run.files()
    try:
        model_files = [e for e in run_files if e.name.endswith('.model')]
    except TypeError:
        print("Found no model files. Exiting.")
        return
    if len(model_files) == 0:
        print("Found no model files. Exiting.")
        return

    if frisbee.args.checkpoint != -1:
        model_files = [e for e in model_files if int(e.name.rstrip('.model').split("_")[-1]) == frisbee.args.checkpoint]

    for file in natsorted(model_files, key=lambda e: e.name):
        file_steps = int(file.name.rstrip('.model').split("_")[-1])
        print(f"Evaluating checkpoint at {file_steps} steps")

        # Set up the evaluation trackers and add them to the frisbee
        add_sns_to_frisbee(setup_evaluation_trackers(frisbee), frisbee)

        # Download the model and restore it
        model = file.download(replace=True, root=wandb.run.dir)
        frisbee.policy_net.load_state_dict(torch.load(model.name, map_location=frisbee.device))
        frisbee.checkpoint_at = file_steps

        # Do evaluation
        frisbee.visualize = False
        metric = evaluate(frisbee)

        if metric >= best_metric:
            best_model_file = file
            best_model = file_steps
            best_metric = metric
            wandb.log({'eval/best_model': best_model,
                       'eval/best_return': best_metric},
                      step=frisbee.checkpoint_at)

    # Visualize the best model
    # Set up the evaluation trackers and add them to the frisbee
    add_sns_to_frisbee(setup_evaluation_trackers(frisbee), frisbee)
    # Download the model and restore it
    model = best_model_file.download(replace=True, root=wandb.run.dir)
    frisbee.policy_net.load_state_dict(torch.load(model.name, map_location=frisbee.device))
    frisbee.checkpoint_at = best_model
    # Evaluate
    frisbee.visualize = True
    evaluate(frisbee)


def main(args):
    # Create a frisbee -- a SimpleNamespace to toss around that contains general information
    frisbee = create_evaluation_frisbee(args)

    # Set up the environment and add it to the frisbee
    add_sns_to_frisbee(setup_env(frisbee), frisbee)

    # Set up the model and add it to the frisbee
    add_sns_to_frisbee(setup_model(frisbee), frisbee)

    # Set up the evaluation trackers and add them to the frisbee
    add_sns_to_frisbee(setup_evaluation_trackers(frisbee), frisbee)

    # Set up the predicates and add them to the frisbee
    add_sns_to_frisbee(setup_predicates(frisbee), frisbee)

    # Put the policy network into evaluation mode
    frisbee.policy_net = frisbee.policy_net.eval()
    torch.set_grad_enabled(False)

    # Do evaluation
    if frisbee.background:
        background_evaluation(frisbee)
    else:
        post_evaluation(frisbee)


if __name__ == '__main__':
    # Set up the argument parser
    parser = ArgumentParser('Pass in arguments!')
    parser.add_argument('--config', '-c', help='Path to the config .yaml file.', type=str, required=True)
    parser.add_argument('--background', '-b', help='Keep evaluation running as a background process.',
                        action='store_true')
    parser.add_argument('--checkpoint', '-p', help='Specify a checkpoint to load and evaluate.', type=int, default=-1)
    args = parser.parse_args()

    # Run!
    main(args)