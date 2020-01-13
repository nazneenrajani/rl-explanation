import matplotlib.pyplot as plt
import torch


def plot_episodic_variables(variables, label, fig_idx):
    plt.figure(fig_idx)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel(label)
    plt.plot(variables.numpy())
    # Take 100 episode averages and plot them too
    if len(variables) >= 100:
        means = variables.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
