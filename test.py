import metaworld

ml10 = metaworld.MT10()
env = ml10.train_classes['reach-v2']()
task = [t for t in ml10.train_tasks if t.env_name == 'reach-v2'][0]
env.set_task(task)

obs, _ = env.reset()

achieved_goal = env.get_endeff_pos()
desired_goal = env._get_pos_goal()

print("Achieved goal:", achieved_goal)
print("Desired goal:", desired_goal)

reward = env.compute_reward(achieved_goal, desired_goal)
print("Reward:", reward)