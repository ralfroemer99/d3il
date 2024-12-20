import sys, os
import logging
import hydra
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

exp = 'avoiding'

ee_position_indices = {
    'avoiding': [2, 3],
    'aligning': [3, 4, 5],
    'pushing': [2, 3],
    'sorting': [2, 3],
    'stacking': [], # 0 to 6: desired joint positions, 7: desired gripper width
}

# Default arguments
config_name = f"{exp}_config"

# Default arguments
default_args = [
    "run.py",
    f"--config-name={config_name}",      # dynamically set config-name
]

# Use default arguments if none are provided
if len(sys.argv) == 1:  # If no arguments are passed
    sys.argv = default_args

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()

# Disable wandb logging
os.environ["WANDB_MODE"] = "disabled"

@hydra.main(config_path="configs", config_name="avoiding_config.yaml")
def main(cfg: DictConfig) -> None:
    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.group,
            mode="disabled",
            config=wandb.config
        )

    agent = hydra.utils.instantiate(cfg.agents)
    
    if exp == 'avoiding':
        observations = agent.trainset.observations
        actions = agent.trainset.actions
    else:
        observations = torch.cat((agent.trainset.observations, agent.valset.observations))
        actions = torch.cat((agent.trainset.actions, agent.valset.actions))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if exp in ['aligning', 'pushing', 'sorting']:
        for _ in range(observations.shape[0]):
            ax.plot(observations[_, :, ee_position_indices[exp][0]], observations[_, :, ee_position_indices[exp][1]])

    plt.show()

if __name__ == "__main__":
    main()