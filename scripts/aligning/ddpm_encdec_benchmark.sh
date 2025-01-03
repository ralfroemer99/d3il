# n_cores: 5, n_contexts: 60, n_trajectories_per_context: 8, seed: 0,1,2,3,4,5
python run.py --config-name=aligning_config \
              --multirun seed=0 \
              agents=ddpm_encdec_agent \
              agent_name=ddpm_encdec \
              window_size=10 \
              group=ddpm_encdec_h1a10 \
              agents.action_seq_size=10 \
              agents.obs_seq_len=1 \
              agents.model.model.action_seq_len=10 \
              agents.model.model.obs_seq_len=1