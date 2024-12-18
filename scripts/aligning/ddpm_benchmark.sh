# n_cores: 5, n_contexts: 60, n_trajectories_per_context: 8, seed: 0,1,2,3,4,5
python run.py --config-name=aligning_config \
              --multirun seed=0 \
              agents=ddpm_agent \
              agent_name=ddpm \
              window_size=1 \
              group=aligning_ddpm_seeds_0.25data \
              simulation.n_cores=1 \
              simulation.n_contexts=1 \
              simulation.n_trajectories_per_context=8 \
              agents.model.model.t_dim=8 \
              agents.model.n_timesteps=24