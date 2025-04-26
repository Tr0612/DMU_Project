import time

def evaluate(action_getter, env, num_episodes=5, render=True):
    if render:
        env.set_render_mode("human")
    # Reset and get initial observation
    obs, _ = env.reset()
    total_successes = 0
    attempts_by_task = {}
    successes_by_task = {}
    for episode in range(num_episodes):
        print(f"\n🎬 Episode {episode + 1}")

        obs, _ = env.reset()
        success = False
        current_task = None
        for eval_step in range(1000):  # 1000 steps max
            if render:
                env.render()

            action = action_getter(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if eval_step == 0:
                print("Task: ", info["task_name"])
                current_task = info["task_name"]
                if current_task != None:
                    attempts_by_task[current_task] = (
                        attempts_by_task.get(current_task, 0) + 1
                    )
            if int(info["success"]) == 1:
                success = True
                print("Success!")
                if render:
                    time.sleep(3)  # Slow down for human viewing
                break
            if terminated:
                break
        if success:
            total_successes += 1
            if current_task:
                successes_by_task[current_task] = (
                    successes_by_task.get(current_task, 0) + 1
                )

    # Show result
    if render:
        time.sleep(3)
    env.close()
    print("Attempts by task: ", attempts_by_task)
    print("Total successes: ", total_successes)
    print("Successes by task: ", successes_by_task)
    return attempts_by_task, successes_by_task
