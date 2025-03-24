import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import pandas as pd

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

cmap = ["r", "g", "b", "y", "c"]
sns.set("notebook", font_scale=1.1, rc={"figure.figsize": (8, 3)})
sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})


data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=10, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=4, # Total number of different scenes we want to use
    sample_with_replacement=True, 
    seed=42, 
    shuffle=True,   
)

# Pass the data_loader to the environment 
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    num_worlds=2,
    max_cont_agents=64,  # Maximum number of agents to control per scene
    data_loader=data_loader,
    device="cuda",
)

# Test if we are loading the dataset properly
print("Testing if we loaded the dataset properly", data_loader.dataset)

# Take an example scene
data_path = data_loader.dataset[0]

with open(data_path) as file:
    traffic_scene = json.load(file)

print("traffic keys: ", traffic_scene.keys())
run.config {'vec': 'multiprocessing', 'mode': 'sweep', 'sweep': {'name': 'sweep', 'method': 'bayes', 'metric': {'goal': 'maximize', 'name': 'environment/episode_return'}, 'parameters': {'train': {'parameters': {'gamma': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'vf_coef': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'ent_coef': {'max': 0.1, 'min': 1e-05, 'distribution': 'log_uniform_values'}, 'gae_lambda': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'bptt_horizon': {'values': [1, 2, 4, 8, 16]}, 'learning_rate': {'max': 0.1, 'min': 1e-05, 'distribution': 'log_uniform_values'}, 'max_grad_norm': {'max': 10, 'min': 0, 'distribution': 'uniform'}, 'update_epochs': {'max': 4, 'min': 1, 'distribution': 'int_uniform'}}}}}, 'track': True, 'train': {'seed': 1, 'gamma': 0.2234618822276343, 'device': 'cpu', 'compile': False, 'vf_coef': 0.19233060413596592, 'data_dir': 'experiments', 'ent_coef': 2.6581454835296473e-05, 'norm_adv': True, 'num_envs': 4, 'anneal_lr': True, 'clip_coef': 0.1, 'target_kl': None, 'zero_copy': True, 'batch_size': 4096, 'clip_vloss': True, 'gae_lambda': 0.7229277193192272, 'cpu_offload': False, 'num_workers': 4, 'bptt_horizon': 16, 'compile_mode': 'reduce-overhead', 'vf_clip_coef': 0.1, 'learning_rate': 1.5155550453076993e-05, 'max_grad_norm': 9.534580890087382, 'update_epochs': 2, 'env_batch_size': 4, 'minibatch_size': 2048, 'total_timesteps': 10000000, 'checkpoint_interval': 200, 'torch_deterministic': True}, 'exp_id': 'test-mmo-sweep-2', 'package': 'nmmo', 'baseline': False, 'env_name': 'nmmo', 'rnn_name': None, 'policy_name': 'Policy', 'render_mode': 'auto', 'wandb_group': 'debug', 'vec_overwork': False, 'wandb_project': 'pufferlib', 'eval_model_path': None, 'max_suggestion_cost': 3600}




import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import pandas as pd

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

cmap = ["r", "g", "b", "y", "c"]
sns.set("notebook", font_scale=1.1, rc={"figure.figsize": (8, 3)})
sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})


data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=10, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=4, # Total number of different scenes we want to use
    sample_with_replacement=True, 
    seed=42, 
    shuffle=True,   
)

# Pass the data_loader to the environment 
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    num_worlds=2,
    max_cont_agents=64,  # Maximum number of agents to control per scene
    data_loader=data_loader,
    device="cuda",
)

# Test if we are loading the dataset properly
print("Testing if we loaded the dataset properly", data_loader.dataset)

# Take an example scene
data_path = data_loader.dataset[0]

with open(data_path) as file:
    traffic_scene = json.load(file)

print("traffic keys: ", traffic_scene.keys())
run.config {'vec': 'multiprocessing', 'mode': 'sweep', 'sweep': {'name': 'sweep', 'method': 'bayes', 'metric': {'goal': 'maximize', 'name': 'environment/episode_return'}, 'parameters': {'train': {'parameters': {'gamma': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'vf_coef': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'ent_coef': {'max': 0.1, 'min': 1e-05, 'distribution': 'log_uniform_values'}, 'gae_lambda': {'max': 1, 'min': 0, 'distribution': 'uniform'}, 'bptt_horizon': {'values': [1, 2, 4, 8, 16]}, 'learning_rate': {'max': 0.1, 'min': 1e-05, 'distribution': 'log_uniform_values'}, 'max_grad_norm': {'max': 10, 'min': 0, 'distribution': 'uniform'}, 'update_epochs': {'max': 4, 'min': 1, 'distribution': 'int_uniform'}}}}}, 'track': True, 'train': {'seed': 1, 'gamma': 0.2234618822276343, 'device': 'cpu', 'compile': False, 'vf_coef': 0.19233060413596592, 'data_dir': 'experiments', 'ent_coef': 2.6581454835296473e-05, 'norm_adv': True, 'num_envs': 4, 'anneal_lr': True, 'clip_coef': 0.1, 'target_kl': None, 'zero_copy': True, 'batch_size': 4096, 'clip_vloss': True, 'gae_lambda': 0.7229277193192272, 'cpu_offload': False, 'num_workers': 4, 'bptt_horizon': 16, 'compile_mode': 'reduce-overhead', 'vf_clip_coef': 0.1, 'learning_rate': 1.5155550453076993e-05, 'max_grad_norm': 9.534580890087382, 'update_epochs': 2, 'env_batch_size': 4, 'minibatch_size': 2048, 'total_timesteps': 10000000, 'checkpoint_interval': 200, 'torch_deterministic': True}, 'exp_id': 'test-mmo-sweep-2', 'package': 'nmmo', 'baseline': False, 'env_name': 'nmmo', 'rnn_name': None, 'policy_name': 'Policy', 'render_mode': 'auto', 'wandb_group': 'debug', 'vec_overwork': False, 'wandb_project': 'pufferlib', 'eval_model_path': None, 'max_suggestion_cost': 3600}




import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import pandas as pd

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

cmap = ["r", "g", "b", "y", "c"]
sns.set("notebook", font_scale=1.1, rc={"figure.figsize": (8, 3)})
sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})


data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=10, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=4, # Total number of different scenes we want to use
    sample_with_replacement=True, 
    seed=42, 
    shuffle=True,   
)

# Pass the data_loader to the environment 
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    num_worlds=2,
    max_cont_agents=64,  # Maximum number of agents to control per scene
    data_loader=data_loader,
    device="cuda",
)

# Test if we are loading the dataset properly
print("Testing if we loaded the dataset properly", data_loader.dataset)

# Take an example scene
data_path = data_loader.dataset[0]

with open(data_path) as file:
    traffic_scene = json.load(file)

print("traffic keys: ", traffic_scene.keys())