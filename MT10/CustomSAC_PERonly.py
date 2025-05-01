from stable_baselines3 import SAC
import torch
import torch.nn.functional as F
from utils import soft_update


class CustomSAC(SAC):
    """
    SAC Agent that supports Prioritized Experience Replay (PER)
    """
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override the standard SAC train() to support PER.
        """
        # Anneal beta if you want (optional)
        beta = 0.4  # Can slowly anneal beta to 1.0 during training

        for gradient_step in range(gradient_steps):
            
            replay_data,indices,weights = self.replay_buffer.sample(batch_size)
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
                if next_log_prob.dim() == 1:
                    next_log_prob = next_log_prob.unsqueeze(1)
                # print(type(next_actions),next_actions.shape)
                # print(type(next_log_prob),next_log_prob.shape)
                # next_q_values = torch.minimum(
                #     self.critic_target.q_net1(replay_data.next_observations, next_actions),
                #     self.critic_target.q_net2(replay_data.next_observations, next_actions),
                # )
                # rewards = replay_data.rewards.squeeze(-1)
                # dones = replay_data.dones.squeeze(-1)
                device = replay_data.dones.device  # Get the device (cpu or cuda)
                one_tensor = torch.ones_like(dones,device=device)
                q1_target, q2_target = self.critic_target(replay_data.next_observations, next_actions)
                next_q_values = torch.min(q1_target, q2_target)
                # print(type(rewards),rewards.shape)
                # print(type(one_tensor),one_tensor.shape)
                # print(type(dones),dones.shape)
                # print(type(next_q_values),next_q_values.shape)
                # print(type(self.ent_coef),self.ent_coef)
                # print(type(next_log_prob),next_log_prob.shape)
                # print(type(self.gamma),self.gamma.shape)
                target_q_values = rewards + (one_tensor - dones) * self.gamma * (next_q_values - self.ent_coef * next_log_prob)

            # # Get current Q estimates
            # current_q1 = self.critic.q_net1(replay_data.observations, replay_data.actions)
            # current_q2 = self.critic.q_net2(replay_data.observations, replay_data.actions)
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            
            # Compute TD errors
            td_error1 = torch.abs(current_q1 - target_q_values)
            td_error2 = torch.abs(current_q2 - target_q_values)
            td_errors = (td_error1 + td_error2) / 2.0  # average two critics

            #  Update Priorities
            new_priorities = td_errors.detach().cpu().numpy().squeeze()
            self.replay_buffer.update_priorities(indices, new_priorities)

            # Critic loss
            critic_loss = (F.mse_loss(current_q1, target_q_values, reduction='none') * weights).mean() \
            + (F.mse_loss(current_q2, target_q_values, reduction='none') * weights).mean()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Actor update
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            
            if log_prob.dim() == 1:
                log_prob = log_prob.unsqueeze(-1)
                
            q_pi1, q_pi2 = self.critic(replay_data.observations, actions_pi)
            q_pi = torch.min(q_pi1,q_pi2)
            actor_loss = (self.ent_coef * log_prob - q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Temperature update
            if getattr(self,"automatic_entropy_tuning",False):
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

                self.ent_coef = self.log_ent_coef.exp()
            else:
                ent_coef_loss = torch.tensor(0.0)

            # Target networks soft update
            if gradient_step % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                # self.soft_update(self.critic_target, self.critic, self.tau)
