import os
import numpy as np
import matplotlib.pyplot as plt
import metaworld
from CustomSAC import CustomSAC

# Evaluation settings
save_dir = "./mt1_models/"
episodes = 1  # reduce for speed; you had 5000
success_rates = {}

# ─── Safe reset & step helpers ───────────────────────────────
def safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out

# ─── Evaluation loop ─────────────────────────────────────────
for filename in os.listdir(save_dir):
    if not filename.endswith(".zip"):
        continue

    model_path = os.path.join(save_dir, filename)
    model_label = filename.replace(".zip", "")

    # Parse task name from filename
    parts = filename.split("_")
    if len(parts) < 3:
        print(f"⚠️ Skipping malformed filename: {filename}")
        continue

    task_name = "_".join(parts[1:-2]) if parts[-2] in ["per", "vanilla"] else "_".join(parts[1:-1])
    try:
        mt1 = metaworld.MT1(task_name, seed=42)
        env = mt1.train_classes[task_name]()
        env.set_task(mt1.train_tasks[0])
    except Exception as e:
        print(f"❌ Could not load environment for {task_name}: {e}")
        continue

    try:
        model = CustomSAC.load(model_path)
    except Exception as e:
        print(f"❌ Could not load model {filename}: {e}")
        continue

    # Evaluate
    print(f"\n🔍 Evaluating {model_label}")
    success = 0
    for ep in range(episodes):
        obs = safe_reset(env)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = safe_step(env, action)
            if info.get("success", 0) == 1:
                success += 1
                break

    rate = success / episodes * 100
    success_rates[model_label] = rate
    print(f"✅ {model_label} success rate: {rate:.1f}%")

# ─── Plotting ────────────────────────────────────────────────
labels = list(success_rates.keys())
rates  = [success_rates[l] for l in labels]

x = np.arange(len(labels))
width = 0.6

fig, ax = plt.subplots(figsize=(max(10, len(labels)), 6))
bars = ax.bar(x, rates, width)

ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate per Model (Filename)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylim(0, 100)
ax.grid(True, axis='y')

# Add value labels
for r in bars:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "success_rate_by_filename.png"))
plt.show()
