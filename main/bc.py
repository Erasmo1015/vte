from typing import Dict, Union

import numpy as np
import torch

class BehaviorCloningLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    def __init__(self, ent_weight: float, l2_weight: float, kld_weight: float, agent_name: str) -> None:
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight
        self.kld_weight = kld_weight
        self.beta = 4
        self.gamma = 1000
        self.loss_type = "B"
        self.C_max = torch.Tensor([25])
        self.C_stop_iter = 1e5
        self.num_iter = 0
        self.agent_name = agent_name

    def pure_bc(
        self,
        agent,
        obs,
        acts: Union[torch.Tensor, np.ndarray],
    ) -> Dict:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        (_, log_prob, entropy) = agent.evaluate_actions(
            obs,  # type: ignore[arg-type]
            acts,
        )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in agent.actor.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (
            entropy if entropy is not None else torch.zeros(1)
        )
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss
        return {
            "bc_actor_loss/neglogp": neglogp,
            "bc_actor_loss/entropy": entropy,
            "bc_actor_loss/ent_loss": ent_loss,
            "bc_actor_loss/prob_true_act": prob_true_act,
            "bc_actor_loss/l2_norm": l2_norm,
            "bc_actor_loss/l2_loss": l2_loss,
            "loss/bc_actor": loss,
        }

    def __call__(
        self,
        agent,
        obs,
        acts: Union[torch.Tensor, np.ndarray],
        mu,
        log_var,
    ) -> Dict:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        if mu is None or log_var is None:
            return self.pure_bc(agent, obs, acts)
        if self.agent_name == "sac":
            (_, log_prob, entropy) = agent.evaluate_actions(
                obs,  # type: ignore[arg-type]
                acts,
            )
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            entropy = entropy.mean() if entropy is not None else None

            l2_norms = [torch.sum(torch.square(w)) for w in agent.actor.parameters()]
            l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
            # sum of list defaults to float(0) if len == 0.
            assert isinstance(l2_norm, torch.Tensor)

            ent_loss = -self.ent_weight * (
                entropy if entropy is not None else torch.zeros(1)
            )
            neglogp = -log_prob
            l2_loss = self.l2_weight * l2_norm
            bc_loss = neglogp + ent_loss + l2_loss
        else: # softq version
            (_, log_prob, entropy) = agent.evaluate_actions(
                obs,  # type: ignore[arg-type]
                acts,
            )
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            entropy = entropy.mean() if entropy is not None else None

            l2_norms = [torch.sum(torch.square(w)) for w in agent.actor.parameters()]
            l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
            # sum of list defaults to float(0) if len == 0.
            assert isinstance(l2_norm, torch.Tensor)

            ent_loss = -self.ent_weight * (
                entropy if entropy is not None else torch.zeros(1)
            )
            neglogp = -log_prob
            l2_loss = self.l2_weight * l2_norm
            bc_loss = neglogp + ent_loss + l2_loss
        result = self.vae_loss_function(bc_loss, mu, log_var) # VAE loss
        # result = self.betaVAE_loss_function(bc_loss, mu, log_var) # beta-VAE loss
        kld_loss = result['KLD_loss']
        loss = result['loss']
        return {
            "bc_actor_loss/neglogp": neglogp,
            "bc_actor_loss/entropy": entropy,
            "bc_actor_loss/ent_loss": ent_loss,
            "bc_actor_loss/prob_true_act": prob_true_act,
            "bc_actor_loss/l2_norm": l2_norm,
            "bc_actor_loss/l2_loss": l2_loss,
            "loss/bc_actor": loss,
            "loss/kld_loss": kld_loss,
        }
    
    def vae_loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args: recons, input, mu, log_var
        :param kwargs:
        :return:
        """
        recons_loss = args[0]
        mu = args[1]
        log_var = args[2]
        kld_weight = self.kld_weight
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = torch.mean(kld_loss)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD_loss':kld_loss.detach()}

    def betaVAE_loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons_loss = args[0]
        mu = args[1]
        log_var = args[2]
        kld_weight = self.kld_weight
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = torch.mean(kld_loss)
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(mu.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD_loss':kld_loss.detach()}
