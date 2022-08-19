#docker run -d -v /root/masterthesis/logs:/root/code/rl_zoo/logs evaluation:0.1 python -u train.py --algo sac --env zinc-coating-v0 --env-kwargs coating_reward_time_offset:${@}
python -u train.py --algo sac --env zinc-coating-v0 -n 50000 --n-trials 1000 --n-jobs 2 -f logs_hyperparameters --optimize --study-name undelayed --storage sqlite:///hyperparameters.db
