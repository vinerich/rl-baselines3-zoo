docker run -d -v /root/masterthesis/logs_easy:/root/code/rl_zoo/logs evaluation:0.3 python -u train.py --algo sac --env zinc-coating-v0 --env-kwargs use_randomized_coil_targets:True
