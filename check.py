from metaworld import MT1
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy as Policy
from evaluate import evaluate

# Initialize MT1 with reach-v2 task
mt1 = MT1('reach-v2', seed=42)
env = mt1.train_classes['reach-v2']()
env.set_task(mt1.train_tasks[0])
env._render_mode = "human"
env.render_mode = "human"

# Load the built-in expert policy
policy = Policy()

evaluate(lambda obs : policy.get_action(obs), env, render=True)
