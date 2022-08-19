from zinc_coating_environment_IS import ZincCoatingV0_IS

env = ZincCoatingV0_IS(coating_reward_time_offset=5)
env.reset()

for i in range(0, 100):
    obs, reward, done, real_coating = env.step([i])
    print(f"obs: {obs}, real: {real_coating}")
    # print(reward)
    # base.reset()
