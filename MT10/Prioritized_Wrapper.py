import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer,ReplayBufferSamples
import torch as th
import torch
from typing import NamedTuple


# class PrioritizedReplayBufferSamples(NamedTuple):
#     observations: th.Tensor
#     actions: th.Tensor
#     next_observations: th.Tensor
#     dones: th.Tensor
#     rewards: th.Tensor
#     indices: th.Tensor
#     weights: th.Tensor

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1,alpha=0.6,**kwargs):
        super().__init__(buffer_size, observation_space, action_space, device,n_envs=n_envs,**kwargs)
        self.alpha = alpha
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        idx = self.pos  # where the sample will go
        super().add(obs, next_obs, action, reward, done, infos)
        max_priority = self.priorities.max() if self.full else self.priorities[:self.pos].max()
        if max_priority == 0:
            max_priority = 1.0  # Initial priority
        self.priorities[idx] = max_priority
    
    def update_priorities(self,indices,new_prios):
        self.priorities[indices] = new_prios

    def sample(self, batch_size, env=None):
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Compute probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(probs), batch_size, p=probs)

        # Compute importance-sampling weights
        weights = (len(probs) * probs[indices]) ** (-1)
        weights /= weights.max()  # Normalize to [0, 1]
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]
        
        # Sample transitions
        replay_data = super().sample(batch_size, env)
        
        return replay_data,indices,weights

        # Save indices and weights internally for later
        # self._last_sampled_indices = indices
        # self._last_sampled_weights = weights
        # device = replay_data.observations.device

        # return PrioritizedReplayBufferSamples(
        #     observations=replay_data.observations.to(device),
        #     actions=replay_data.actions.to(device),
        #     next_observations=replay_data.next_observations.to(device),
        #     dones=replay_data.dones.to(device),
        #     rewards=replay_data.rewards.to(device),
        #     indices=torch.tensor(indices, device=device),
        #     weights=torch.tensor(weights, device=device).unsqueeze(1),
# )
        # return ReplayBufferSamples(
        #     observations=replay_data.observations,
        #     actions=replay_data.actions,
        #     next_observations=replay_data.next_observations,
        #     dones=replay_data.dones,
        #     rewards=replay_data.rewards,
        # )
        


    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


