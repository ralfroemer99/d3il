# Seeds: 0,1,2,3,4,5 n_cores: 30, n_trajectories: 480, window_size: 8
python run.py --config-name=avoiding_config \
              --multirun seed=0 \
              agents=ddpm_encdec_agent \
              agent_name=ddpm_encdec \
              window_size=8 \
              group=avoiding_ddpm_encdec_seeds \
              simulation.n_cores=1 \
              simulation.n_trajectories=10 \
              agents.model.n_timesteps=16