from metaworld import MT1
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy as Policy
import time

# Initialize MT1 with reach-v2 task
mt1 = MT1('reach-v2', seed=42)
env = mt1.train_classes['reach-v2']()
env.set_task(mt1.train_tasks[0])
env._render_mode = "human"
env.render_mode = "human"
# Reset and get initial observation
obs, _ = env.reset()

# Load the built-in expert policy
policy = Policy()

# Run one episode with the policy
success = False

num_episodes = 5
for episode in range(num_episodes):
    print(f"\n🎬 Episode {episode + 1}")

    obs, _ = env.reset()
    success = False
    for _ in range(200):  # 200 steps max
        env.render()

        action = policy.get_action(obs)
        obs, _, terminated, truncated, info = env.step(action)

        if int(info['success']) == 1:
            success = True
            
            # time.sleep(100)  # Slow down for human viewing

# Show result

time.sleep(3)
env.close()
