# Explanations in Reinforcement Learning
Repository for code and experiments related to the RLExp project.

#### Set up the Conda environment

``conda create -n rlexp python=3.6``

``conda install pytorch torchvision -c pytorch``

``pip install -r requirements.txt``

#### Docker env
`cd` into the `/gcp` folder and run

`gcloud builds submit --tag gcr.io/salesforce-research-internal/karan-goel-image-rlexp . --timeout 1800`

#### Running 

The files for running training and evaluation are,

- `horde.py` (Phase 1: DQN training)
- `offline_horde.py` (Phase 2: Learning aux Q fns)
- `distill_horde.py` (Phase 3: Distilling the Q fn by regressing from aux Qs)
- `evaluate_horde.py` (Phase 4: Evaluating the learned models)

Each phase has an associate configuration template that can be found in `config/template_{filename}`. The template
contains defaults for each option. If an option is not set, the default in the template is used.

Running a phase is done by creating a `.yaml` configuration file (copy and edit the corresponding template), 
and then running,

``<filename.py> -c config.yaml``

To launch jobs on GCP using Kubernetes, use the `run_job.py` file, and pass in a sweep configuration file as input. 
An example can be found at `sweep_configs/example.yaml`. These sweep configs take multiple hyperparameter settings as 
input and run all cross combinations of those hyper-parameter settings as a separate Kubernetes job.
