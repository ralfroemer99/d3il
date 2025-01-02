import sys
import logging
import random

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch

TRAIN = False
SIM = True

# Default arguments
default_args = [
    "run.py",
    # "--config-name=aligning_config",            # aligning/home/ralf/Projects/d3il/logs/avoiding/sweeps/ddpm_encdec/2025-01-02/21-39-17/action_dim=2,action_space=pos,agent_name=ddpm_encdec,agents.action_seq_size=4,agents.model.model.action_seq_len=8,agents.model.model.obs_seq_len=1,agents.obs_seq_len=1,agents=ddpm_encdec_agent,group=avoiding_ddpm_encdec,seed=0,window_size=8/home/ralf/Projects/d3il/logs/avoiding/sweeps/ddpm_encdec/2025-01-02/21-39-17/action_dim=2,action_space=pos,agent_name=ddpm_encdec,agents.action_seq_size=4,agents.model.model.action_seq_len=8,agents.model.model.obs_seq_len=1,agents.obs_seq_len=1,agents=ddpm_encdec_agent,group=avoiding_ddpm_encdec,seed=0,window_size=8
    # "--multirun",
    # "window_size=10",
    # "group=ddpm_encdec",
    # "agents.action_seq_size=10",
    # "agents.obs_seq_len=1",
    # "agents.model.model.action_seq_len=10",
    # "agents.model.model.obs_seq_len=1",
    # "--config-name=pushing_config",             # pushing
    # "--multirun",
    # "window_size=8",
    # "group=aligning_ddpm_encdec",
    "--config-name=avoiding_config",          # avoiding
    "--multirun",
    "window_size=8",                          # obs_seq_len + act_seq_len - 1
    "group=avoiding_ddpm_encdec",
    "agents.model.model.action_seq_len=8",
    "agents.model.model.obs_seq_len=1",
    "agents.obs_seq_len=1",
    "agents.action_seq_size=4",
    # "agents.optimization.lr=1e-4",
    "action_dim=2",
    "action_space=pos",
    # For all
    "seed=0",
    "agents=ddpm_encdec_agent",
    "agent_name=ddpm_encdec",
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
            agent.load_pretrained_model('/home/ralf/Projects/d3il/logs/avoiding/sweeps/ddpm_encdec/2025-01-02/21-39-17/action_dim=2,action_space=pos,agent_name=ddpm_encdec,agents.action_seq_size=4,agents.model.model.action_seq_len=8,agents.model.model.obs_seq_len=1,agents.obs_seq_len=1,agents=ddpm_encdec_agent,group=avoiding_ddpm_encdec,seed=0,window_size=8', 
                                        sv_name=agent.eval_model_name)

        # simulate the model
        env_sim = hydra.utils.instantiate(cfg.simulation)
        env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()