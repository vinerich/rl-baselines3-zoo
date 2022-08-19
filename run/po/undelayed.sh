#docker run -d -v /root/masterthesis/logs:/root/code/rl_zoo/logs evaluation:0.1 python -u train.py --algo sac --env zinc-coating-v0 --env-kwargs coating_reward_time_offset:${@}
python -u train.py --algo sac --env zinc-coating-po-v0 -n 300000 -f logs/po/d0
