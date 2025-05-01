from stable_baselines3 import SAC
import torch
import torch as th
import torch.nn.functional as F
from utils import soft_update
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import numpy as np


class CustomSAC(SAC):
    """
    SAC Agent that supports Prioritized Experience Replay (PER)
    """
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override the standard SAC train() to support PER.
        """
        
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        
        # Anneal beta if you want (optional)
        beta = 0.4  # Can slowly anneal beta to 1.0 during training

        for gradient_step in range(gradient_steps):
            
            sampled = self.replay_buffer.sample(batch_size)
            
            if isinstance(sampled, tuple) and len(sampled) == 3:
                # PER: (ReplayBufferSamples, indices, weights)
                replay_data, indices, weights = sampled
            else:
                # vanilla: just ReplayBufferSamples
                replay_data = sampled
                indices = None
                weights = None
            obs      = replay_data.observations
            actions  = replay_data.actions
            next_obs = replay_data.next_observations
            rewards  = replay_data.rewards
            dones    = replay_data.dones
            # print("=== replay_data structure ===")
            # print(type(replay_data))
            # print(replay_data)
            # print("replay_data.rewards:", type(replay_data.rewards))
            # print("replay_data.dones:", type(replay_data.dones))
            # print("replay_data.next_observations:", type(replay_data.next_observations))
            # print("replay_data.actions:", type(replay_data.actions))
            # # print(type(replay_data.rewards), replay_data.rewards.shape)
            # print(type(replay_data.dones), replay_data.dones.shape)
            # Compute critic loss
            with torch.no_grad():
                # Target actions come from the target policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                actions_pi = next_actions
                log_prob = next_log_prob
                if next_log_prob.dim() == 1:
                    next_log_prob = next_log_prob.unsqueeze(1)
                    
                ent_coef = th.exp(self.log_ent_coef.detach()) if (self.ent_coef_optimizer is not None and self.log_ent_coef is not None) else self.ent_coef_tensor
                
                # print(type(next_actions),next_actions.shape)
                # print(type(next_log_prob),next_log_prob.shape)
                # next_q_values = torch.minimum(
                #     self.critic_target.q_net1(replay_data.next_observations, next_actions),
                #     self.critic_target.q_net2(replay_data.next_observations, next_actions),
                # )
                # rewards = replay_data.rewards.squeeze(-1)
                # dones = replay_data.dones.squeeze(-1)
                one_tensor = torch.ones_like(dones)
                
                q1_target, q2_target = self.critic_target(replay_data.next_observations, next_actions)
                next_q_values = torch.min(q1_target, q2_target)
                # print(type(rewards),rewards.shape)
                # print(type(one_tensor),one_tensor.shape)
                # print(type(dones),dones.shape)
                # print(type(next_q_values),next_q_values.shape)
                # print(type(self.ent_coef),self.ent_coef)
                # print(type(next_log_prob),next_log_prob.shape)
                # print(type(self.gamma),self.gamma.shape)
                target_q_values = rewards + (one_tensor - dones) * self.gamma * (next_q_values - ent_coef * next_log_prob)
            
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            
            if log_prob.dim() == 1:
                log_prob = log_prob.unsqueeze(1)
                
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            # # Get current Q estimates
            # current_q1 = self.critic.q_net1(replay_data.observations, replay_data.actions)
            # current_q2 = self.critic.q_net2(replay_data.observations, replay_data.actions)
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            
            # Compute TD errors
            td_error1 = torch.abs(current_q1 - target_q_values)
            td_error2 = torch.abs(current_q2 - target_q_values)
            td_errors = (td_error1 + td_error2) / 2.0  # average two critics
            # print("TD errors min/max/mean:", td_errors.min().item(), td_errors.max().item(), td_errors.mean().item())

            #  Update Priorities
            if indices is not None:
                new_priorities = td_errors.detach().cpu().numpy().squeeze()
                self.replay_buffer.update_priorities(indices, new_priorities)
                
                if gradient_step % 1000 == 0 and self.replay_buffer.full:
                    priorities = self.replay_buffer.priorities
                    mean_priority = np.mean(priorities)
                    std_priority = np.std(priorities)
                    max_priority = np.max(priorities)

                    self.logger.record("train/per_mean_priority", mean_priority)
                    self.logger.record("train/per_std_priority", std_priority)
                    self.logger.record("train/per_max_priority", max_priority)
            
            if weights is not None:
                critic_loss = (F.mse_loss(current_q1, target_q_values, reduction='none') * weights).mean() \
                + (F.mse_loss(current_q2, target_q_values, reduction='none') * weights).mean()
            
            else:
                # current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * (F.mse_loss(current_q1, target_q_values) + F.mse_loss(current_q2, target_q_values))
            
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Critic loss

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Actor update
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

