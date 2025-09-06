import isaacgym
import isaacgymenvs
import torch

import yaml
from rl_games.algos_torch.players import PpoPlayerContinuous

# Setup paths
path_to_train_config = 'cfg/train/CartpolePPO.yaml'
ckpt_path = "checkpoint=runs/Cartpole_14-23-35-10/nn/Cartpole.pth"

# Make env
num_envs = 2000
envs = isaacgymenvs.make(
	seed=0, 
	task="Cartpole", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless="False",
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

# Load train config
with open(path_to_train_config, "r") as f:
	sim_config = yaml.safe_load(f)
env_info = {
	"observation_space":envs.observation_space,
	"action_space":envs.action_space
	}
sim_config["params"]["config"]["env_info"] = env_info

# Restore model
policy = PpoPlayerContinuous(params=sim_config["params"])
policy.has_batch_dimension = True
policy.restore(ckpt_path)
policy.reset()
obs_dict = envs.reset()

# Simulate
for _ in range(4000):
	actions = policy.get_action(obs=obs_dict["obs"])
	obs_dict, rew_buf, reset_buf, extras = envs.step(actions)
