import time

def evaluate(action_getter, env, num_episodes = 5, render=True):
    # Reset and get initial observation
    obs, _ = env.reset()
    total_successes = 0
    for episode in range(num_episodes):
        print(f"\n🎬 Episode {episode + 1}")

        obs, _ = env.reset()
        success = False
        for _ in range(1000):  # 1000 steps max
            if render:
                env.render()

            action = action_getter(obs)
            obs, _, terminated, truncated, info = env.step(action)

            if int(info['success']) == 1:
                success = True
                print("Success!")
                if render:
                    time.sleep(3)  # Slow down for human viewing
                break
    if success:
        total_successes += 1

    # Show result
    if render:
        time.sleep(3)
    env.close()

    print("Total successes: ", total_successes)
    return total_successes
