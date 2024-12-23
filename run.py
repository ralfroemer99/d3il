import sys
import logging
import random

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch

TRAIN = True
SIM = True

# Default arguments
default_args = [
    "run.py",
    # "--config-name=aligning_config",      # aligning
    # "--multirun",
    # "seed=0",
    # "agents=ddpm_agent",
    # "agent_name=ddpm",
    # "window_size=1",
    # "group=aligning_ddpm_seeds_0.25data",
    # "simulation.n_cores=1",
    # "simulation.n_contexts=2",
    # "simulation.n_trajectories_per_context=8",
    # "agents.model.model.t_dim=8",
    # "agents.model.n_timesteps=24",
    "--config-name=avoiding_config",        # avoiding
    "--multirun",
    "seed=0",
    "window_size=8",
    "agents=ddpm_encdec_agent",
    "agent_name=ddpm_encdec",
    "group=avoiding_ddpm_encdec_seeds",
    "simulation.n_cores=1",
    "simulation.n_trajectories=10",
    "agents.model.n_timesteps=16",
    # "window_size=1",
    # "agents=ddpm_agent",
    # "agent_name=ddpm",
    # "group=avoiding_ddpm_seeds",
    # "simulation.n_cores=1",
    # "simulation.n_trajectories=10",
    # "agents.model.n_timesteps=10",
    "action_dim=2",
    "action_space=vel",
]

# Use default arguments if none are provided
if len(sys.argv) == 1:  # If no arguments are passed
    sys.argv = default_args


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="avoiding_config.yaml")
def main(cfg: DictConfig) -> None:

    # if cfg.seed in [0, 1]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif cfg.seed in [2, 3]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # elif cfg.seed in [4, 5]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    set_seed_everywhere(cfg.seed)

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
    
    # train the agent
    if TRAIN:
        agent.train_agent()

    # load the model performs best on the evaluation set
    if SIM:
        if TRAIN:
            agent.load_pretrained_model(agent.working_dir, sv_name=agent.eval_model_name)
        else:
            agent.load_pretrained_model('/home/ralf/projects/d3il/logs/avoiding/sweeps/ddpm_encdec/2024-12-23/15-25-59/action_dim=2,action_space=vel,agent_name=ddpm_encdec,agents.model.n_timesteps=16,agents=ddpm_encdec_agent,group=avoiding_ddpm_encdec_seeds,seed=0,simulation.n_cores=1,simulation.n_trajectories=10,window_size=8', 
                                        sv_name=agent.eval_model_name)

        # simulate the model
        env_sim = hydra.utils.instantiate(cfg.simulation)
        env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()