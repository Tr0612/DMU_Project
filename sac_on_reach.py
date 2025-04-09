from metaworld import MT1
from evaluate import evaluate
from sac_policy import SACPolicy

# Initialize MT1 with reach-v2 task
mt1 = MT1('reach-v2', seed=42)
env = mt1.train_classes['reach-v2']()
env.set_task(mt1.train_tasks[0])
env._render_mode = "human"
env.render_mode = "human"

# Train a SAC policy
policy = SACPolicy(env)
policy.train_model(timesteps=5000)
evaluate(lambda obs : policy.run_model(obs), env, render=True)