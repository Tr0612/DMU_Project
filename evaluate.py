import time
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


def record_rollout(algo, env, video_path, max_steps=500):
    writer = imageio.get_writer(video_path, fps=30)
    obs, _ = env.reset()
    done, truncated = False, False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        frame = env.render()
        writer.append_data(frame)

        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    writer.close()


def evaluate_withChars(
    checkpoint_path, save_dir="evaluation_results", episodes_per_task=30
):
    os.makedirs(save_dir, exist_ok=True)

    # Load trained model
    # algo = SACAlgorithm.from_checkpoint(checkpoint_path)

    ml10 = metaworld.MT10()
    task_names = list(ml10.train_classes.keys())
    results = {}

    for task_name in task_names:
        print(f"Evaluating task: {task_name}")

        # Create environment
        env = ml10.train_classes[task_name]()
        task = [t for t in ml10.train_tasks if t.env_name == task_name][0]
        env.set_task(task)

        success_count = 0

        for ep in range(episodes_per_task):
            obs, _ = env.reset()
            done, truncated = False, False

            while not (done or truncated):
                action = algo.compute_single_action(obs)
                obs, reward, done, truncated, info = env.step(action)

                if info.get("success", 0.0) == 1.0:
                    success_count += 1
                    break

        success_rate = success_count / episodes_per_task
        results[task_name] = success_rate

        # Record video for each task
        video_path = os.path.join(save_dir, f"{task_name}.mp4")
        record_rollout(algo, env, video_path)

    # Save and plot results
    task_names_sorted = list(results.keys())
    success_rates = [results[name] * 100 for name in task_names_sorted]

    plt.figure(figsize=(12, 6))
    plt.bar(task_names_sorted, success_rates)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Success Rate (%)")
    plt.title("MT10 Task Success Rates")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "success_rates.png"))
    plt.show()

    print("\nFinal Results:")
    for task, score in results.items():
        print(f"{task}: {score*100:.2f}% success rate")


def evaluate(action_getter, env, num_episodes=5, render=True, on_step=None):
    if render:
        env.set_render_mode("human")
    # Reset and get initial observation
    obs, _ = env.reset()
    total_successes = 0
    attempts_by_task = {}
    successes_by_task = {}
    total_steps = 0
    for episode in range(num_episodes):
        print(f"\n🎬 Episode {episode + 1}")
        # if max_steps != None and total_steps > max_steps:
        #     break
        obs, _ = env.reset()
        success = False
        current_task = None
        for eval_step in range(int(5e4)):  # 5000 steps max
            total_steps += 1
            if render:
                env.render()

            action = action_getter(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            if on_step != None:
                on_step(obs, next_obs, action, reward, done, info)
            obs = next_obs
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
            if done:
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
