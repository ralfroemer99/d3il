# Seeds: 0,1,2,3,4,5 n_cores: 10, n_trajectories: 480
python run.py --config-name=avoiding_config \
              --multirun seed=0 \
              agents=ddpm_agent \
              agent_name=ddpm \
              window_size=1 \
              group=avoiding_ddpm_seeds \
              simulation.n_cores=1 \
              simulation.n_trajectories=10 \
              agents.model.model.t_dim=24 \
              agents.model.n_timesteps=4