import numpy as np
import torch
import torch.nn.functional as F


class A2C:
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        critic_weight=0.5,
        entropy_weight=0.01,
    ):
        """
        Initialize A2C algorithm

        Args:
            model: neural network structure (ActorCritic for A2C)
            learning_rate: the learning rate for optimizer
            critic_weight: coefficient for Critic loss
            entropy_weight: coefficient for entropy in the total loss to encourage exploration
        """
        self.model = model
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def learn(
        self,
        batch_states,
        batch_actions,
        batch_returns,
        batch_advantages,
    ):
        """
        Update A2C based on a collected batch

        Args:
            batch_states: Tensor that stores states
            batch_actions: Tensor that stores taken actions
            batch_returns: Tensor that stores the target returns of the Critic
            batch_returns: Tensor that stores the the advantages of the Actor

        Returns:
            Tuple: (total_loss, actor_loss, critic_loss, entropy)
        """

        new_action_log_probs, entropies, new_state_values = self.model.evaluate(batch_states, batch_actions)

        entropy = entropies.mean()

        # Calculate Actor loss
        actor_loss = - (new_action_log_probs * batch_advantages).mean()

        # Calculate Critic loss
        critic_loss = F.mse_loss(new_state_values, batch_returns)

        # Calculate total loss
        loss = (
            actor_loss
            + self.critic_weight * critic_loss
            - self.entropy_weight * entropy
        )

        # Update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
