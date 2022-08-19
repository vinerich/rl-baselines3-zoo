from zinc_coating.base import ZincCoatingBase

base = ZincCoatingBase(coating_reward_time_offset=0, random_coating_targets=True)
base.reset()

for i in range(0, 100):
    obs, reward, real_coating = base.step(i)
    print(f"speed: {obs.coil_speed}, coating: {obs.zinc_coating}, real: {real_coating}")
    # print(reward)
    # base.reset()
