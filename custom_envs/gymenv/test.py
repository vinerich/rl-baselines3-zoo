from zinc_coating.base import ZincCoatingBase

base = ZincCoatingBase(coating_reward_time_offset=10)
base.reset()

for i in range(0, 100):
    print(base.step(300))
