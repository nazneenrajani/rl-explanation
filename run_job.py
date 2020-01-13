import yaml
import wandb
import argparse
from dqn import load_config, get_path_to_rlexp
from types import SimpleNamespace
import itertools
import os
import subprocess


def load_base_config_for_job(all_info):
    file = f'gcp/job_config_{all_info.n_gpus}gpu.yaml'
    with open(file) as f:
        return yaml.load(f)


def create_specialized_job_config(pconfig_store_path, all_info):
    # Load up the base config
    specialized_jconfig = load_base_config_for_job(all_info)

    # Modify it appropriately
    specialized_jconfig['spec']['template']['spec']['containers'][0]['command'] = ['/bin/sh', '-c']

    cmds = ["cd /export/home/rl-explanation/",
            "pip install gym-minigrid",
            "pip install seaborn",
            "pip install natsort",
            "pip install tqdm",
            "pip install --upgrade torch",
            "bash /export/home/.wandb/auth",
            "apt-get -y update",
            "apt-get -y install libxext6 libx11-6 libxrender1 libxtst6 libxi6 libxml2 libglib2.0-0 gdb",
            "eval `ssh-agent -s`",
            "ssh-add /export/home/.ssh/id_rsa",
            "git remote set-url origin git@github.com:MetaMind/rl-explanation.git",
            "git pull",
            f"OMP_NUM_THREADS=1 gdb -ex r -ex backtrace full --args python {all_info.swpconfig.config_for}.py -c {pconfig_store_path}"]
    # f"(python dqn.py -c {pconfig_store_path} &) && python evaluate.py -c {pconfig_store_path}"]
    # " & "]

    specialized_jconfig['spec']['template']['spec']['containers'][0]['args'] = [" && ".join(cmds)]
    specialized_jconfig['metadata']['name'] += f'-{all_info.wandb.id}-{all_info.job_id}'
    specialized_jconfig['spec']['template']['spec']['containers'][0]['name'] += \
        f'-{all_info.wandb.id}-{all_info.job_id}'

    # Store the modified config
    specialized_jconfig_store_path = pconfig_store_path.replace("pconfig", "jconfig")
    yaml.dump(specialized_jconfig, open(specialized_jconfig_store_path, 'w'))

    return specialized_jconfig_store_path


def launch_job(jconfig_store_path):
    # Execute a job which handles everything
    cmd = ["kubectl", "create", "-f", f"{jconfig_store_path}"]
    subprocess.run(cmd)


def determine_sweep_parameters(config):
    swiffers = SimpleNamespace()
    for parameter in config.__dict__:
        if type(config.__dict__[parameter]) == dict and 'values' in config.__dict__[parameter]:
            swiffers.__dict__[parameter] = config.__dict__[parameter]['values']

    return swiffers


def git_push(all_info):
    cmds = [['git', 'add', f'{all_info.rlexp_path}/sweep_configs/*'],
            ['git', 'commit', '-m', 'cfgupdates'],
            ['git', 'push']]
    for cmd in cmds:
        subprocess.run(cmd)


def launch_sweep(args):
    # Create a simple namespace to keep track of everything
    all_info = SimpleNamespace()
    all_info.rlexp_path = get_path_to_rlexp()
    all_info.sweep_config = args.config.rstrip(".yaml")
    all_info.n_gpus = args.n_gpus

    # Load a parameter configuration with sweeps
    all_info.swpconfig = load_config(args.config)

    # Set up wandb
    all_info.wandb = wandb.init(project=f'{all_info.swpconfig.wandb_project}_meta',
                                tags=['sweep'],
                                entity="salesforce",
                                dir=all_info.rlexp_path + "/wandb")
    wandb.config.update(all_info.swpconfig)

    # Determine what the parameters to sweep over are (aka swiffers)
    all_info.swiffers = determine_sweep_parameters(all_info.swpconfig)

    # Generate all possible combinations of the swiffers
    all_parameter_choices = list(itertools.product(*all_info.swiffers.__dict__.values()))

    wandb.log({'Swiffers': wandb.Table(data=all_parameter_choices,
                                       columns=list(all_info.swiffers.__dict__.keys()))}, step=0)

    # Sweep over these parameter configurations
    for i, element in enumerate(all_parameter_choices):
        # Wrap this parameter choice in a namespace
        parameter_choice = SimpleNamespace()
        parameter_choice.__dict__.update(dict(zip(all_info.swiffers.__dict__.keys(), element)))

        # Construct a full parameter config that implements this parameter choice (a swiffed config)
        swiffed_pconfig = SimpleNamespace()
        swiffed_pconfig.__dict__.update(all_info.swpconfig.__dict__)
        for parameter in parameter_choice.__dict__:
            swiffed_pconfig.__dict__[parameter] = parameter_choice.__dict__[parameter]

        swiffed_pconfig.wandb_tags += f"/{all_info.wandb.id}/{i+1}"
        # swiffed_pconfig.wandb_project += f'_{all_info.wandb.id}'

        # Store this swiffed config
        os.makedirs(f'{all_info.sweep_config}_swiffed', exist_ok=True)
        swiffed_pconfig_store_path = f'{all_info.sweep_config}_swiffed/pconfig_{i + 1}.yaml'
        yaml.dump(swiffed_pconfig.__dict__, open(swiffed_pconfig_store_path, 'w'))

        # Create a specialized job config for each parameter setting
        all_info.job_id = i
        specialized_jconfig_store_path = create_specialized_job_config(swiffed_pconfig_store_path, all_info)

        # Push this to remote
        git_push(all_info)

        # Launch a job using this config
        if not args.test:
            launch_job(specialized_jconfig_store_path)


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The sweep configuration file (see sweep_configs/example.yaml for an example) to run.')
    parser.add_argument('--test', '-t', action='store_true', help='Do everything except launching jobs.')
    parser.add_argument('--n_gpus', '-n', type=int, help='Number of GPUs to use in the job. '
                                                         'This will change which job_config yaml is used.', default=4)
    args = parser.parse_args()

    # Launch jobs to implement the parameter sweep
    launch_sweep(args)
