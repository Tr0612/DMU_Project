from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class SACCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=1)

    def _on_step(self):
        # print("Reward: ", self.model.ep_info_buffer)
        return super()._on_step()

class SACPolicy:
    def __init__(self, env):
        self.model = SAC("MlpPolicy", env, verbose=1)
        self.cb = SACCallback()

    def train_model(self, timesteps):
        self.model.learn(
            total_timesteps=timesteps,
            callback=self.cb,
            progress_bar=True)
    
    def run_model(self, next_observation):
        return self.model.predict(next_observation, deterministic=True)[0]
