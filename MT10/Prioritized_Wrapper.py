import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, alpha=0.6, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, **kwargs)
        self.alpha = alpha
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.priorities.max() if self.size() > 0 else 1.0

    def sample(self, batch_size, env=None):
        if self.full:
            probs = self.priorities
        else:
            probs = self.priorities[:self.pos]

        probs = probs ** self.alpha
        
        if probs.sum() == 0:
        # fallback to uniform sampling if all priorities are zero
            probs = np.ones_like(probs)
        
        probs /= probs.sum()

        indices = np.random.choice(len(probs), batch_size, p=probs)
        data = super()._get_samples(indices)
        # data.indices = indices  # Save indices for updating priorities later
        return data,indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-5