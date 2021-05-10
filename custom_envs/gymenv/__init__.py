from gym.envs.registration import register

# register(
#     id='zinc-coating-v0',
#     entry_point='gymenv.zinc_coating_environment:ZincCoatingV0',
# )

register(
    id='zinc-coating-discrete-v0',
    entry_point='gymenv.zinc_coating_discrete:ZincCoatingV0',
)
