from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from scipy.optimize import minimize

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from custom_algos.mpo.policies import MlpPolicy


class MPO(OffPolicyAlgorithm):
    """
    Maximum A Posteriori Policy Optimization (MPO)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)

    :param gamma: the discount factor
    :param dual_constraint: (float) hard constraint of the dual formulation in the E-step
    :param kl_mean_constraint: (float) hard constraint of the mean in the M-step
    :param kl_var_constraint: (float) hard constraint of the covariance in the M-step
    :param alpha: (float) scaling factor of the lagrangian multiplier in the M-step
    :param lagrange_iterations: (int) number of optimization steps of the Lagrangian
    :param action_samples: (int) number of additional actions

    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[MlpPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 5000,
        batch_size: int = 256,
        tau: float = 0.02,
        gamma: float = 0.99,
        dual_constraint: float = 0.1,
        kl_mean_constraint: float = 0.1,
        kl_var_constraint: float = 1e-3,
        alpha: float = 10,
        lagrange_iterations: int = 5,
        action_samples: int = 64,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 4,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 2,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = 0,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(MPO, self).__init__(
            policy,
            env,
            MlpPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_update_interval = target_update_interval

        self.α = alpha  # scaling factor for the update step of η_μ
        self.ε_dual = dual_constraint  # hard constraint for the KL
        self.ε_kl_μ = kl_mean_constraint  # hard constraint for the KL
        self.ε_kl_Σ = kl_var_constraint  # hard constraint for the KL
        self.lagrange_iterations = lagrange_iterations
        self.action_samples = action_samples

        if seed is not None:
            np.random.seed(seed)

        self.η = np.random.rand()
        self.η_kl_μ = 0.0
        self.η_kl_Σ = 0.0
        self.η_kl = 0.0

        self.loss_function = nn.SmoothL1Loss()

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(MPO, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.features_dim = self.actor.features_dim
        self.action_dim = self.actor.action_dim

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        mean_loss_q, mean_loss_p, mean_loss_l, mean_est_q, max_kl_μ, max_kl_Σ, max_kl = [], [], [], [], [], [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            batch_size = replay_data.observations.size(0)

            # Sample "action_samples" num additional actions
            next_action_distribution, _, _ = self.actor_target.get_action_dist(replay_data.next_observations)
            sampled_next_actions = next_action_distribution.sample((self.action_samples,)).transpose(0, 1)

            # Compute mean of q values for the samples
            expanded_next_observations = replay_data.next_observations[:, None, :].expand(-1, self.action_samples, -1)
            sampled_next_actions_expected_q = get_min_critic_tensor(self.critic_target.forward(
                expanded_next_observations.reshape(-1, self.features_dim),
                sampled_next_actions.reshape(-1, self.action_dim)
            )).reshape(batch_size, self.action_samples).mean(dim=1)

            # Compute total expected return
            sampled_expected_return = replay_data.rewards + self.gamma * sampled_next_actions_expected_q

            # Optimize the critic
            critic_q = get_min_critic_tensor(self.critic.forward(replay_data.observations, replay_data.actions)).squeeze()
            critic_loss = self.loss_function(sampled_expected_return, critic_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            critic_losses.append(critic_loss.item())
            mean_est_q.append(critic_q.abs().mean().item())

            # Sample additional actions for E-Step
            with th.no_grad():
                target_action_distribution, target_mean_actions, target_cholesky = self.actor_target.get_action_dist(
                    replay_data.observations)
                sampled_actions = target_action_distribution.sample((self.action_samples,))
            #   for _ in range(self.action_samples):
            #     sampled_action = target_action_distribution.sample()
            #     sampled_actions.append(sampled_action)
            #   sampled_actions = th.tensor(sampled_actions).to(self.device)

                # Compute q values for the samples
                expanded_observations = replay_data.observations[None, ...].expand(self.action_samples, -1, -1)
                sampled_actions_expected_q = get_min_critic_tensor(self.critic_target.forward(
                    expanded_observations.reshape(-1, self.features_dim),
                    sampled_actions.reshape(-1, self.action_dim)
                )).reshape(self.action_samples, batch_size)
                sampled_actions_expected_q_np = sampled_actions_expected_q.cpu().numpy()

            # Define dual function
            def dual(η):
                max_q = np.max(sampled_actions_expected_q_np, 0)
                return η * self.ε_dual + np.mean(max_q) \
                    + η * np.mean(np.log(np.mean(np.exp((sampled_actions_expected_q_np - max_q) / η), axis=0)))

            bounds = [(1e-6, None)]
            res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
            self.η = res.x[0]

            qij = th.softmax(sampled_actions_expected_q / self.η, dim=0)

            # M-Step
            for _ in range(self.lagrange_iterations):
                _, mean_actions, cholesky = self.actor.get_action_dist(replay_data.observations)
                π1 = MultivariateNormal(mean_actions, scale_tril=target_cholesky)
                π2 = MultivariateNormal(target_mean_actions, scale_tril=cholesky)
                loss_p = th.mean(qij * (
                    π1.expand((self.action_samples, batch_size)).log_prob(sampled_actions)
                    + π2.expand((self.action_samples, batch_size)).log_prob(sampled_actions)
                )
                )
                mean_loss_p.append((-loss_p).item())

                kl_μ, kl_Σ = gaussian_kl(
                    μi=target_mean_actions, μ=mean_actions,
                    Ai=target_cholesky, A=cholesky
                )
                max_kl_μ.append(kl_μ.item())
                max_kl_Σ.append(kl_Σ.item())

                self.η_kl_μ -= self.α * (self.ε_kl_μ - kl_μ).detach().item()
                self.η_kl_Σ -= self.α * (self.ε_kl_Σ - kl_Σ).detach().item()

                if self.η_kl_μ < 0.0:
                    self.η_kl_μ = 0.0
                if self.η_kl_Σ < 0.0:
                    self.η_kl_Σ = 0.0

                self.actor.optimizer.zero_grad()
                actor_loss = -(loss_p + self.η_kl_μ * (self.ε_kl_μ - kl_μ)
                                      + self.η_kl_Σ * (self.ε_kl_Σ - kl_Σ)
                               )
                actor_loss.backward()
                clip_grad_norm_(self.actor.parameters(), 0.1)
                self.actor.optimizer.step()

                actor_losses.append(actor_loss.item())

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(MPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(MPO, self)._excluded_save_params() + ["actor", "actor_target", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = []
        # saved_pytorch_variables = ["log_ent_coef"]
        return state_dicts, saved_pytorch_variables


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: mean and covariance terms of the KL
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = th.log(Σ.det() / Σi.det()) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * th.mean(inner_μ)
    C_Σ = 0.5 * th.mean(inner_Σ)
    return C_μ, C_Σ


def get_min_critic_tensor(critics):
    return th.min(th.cat(critics, dim=1), dim=1, keepdim=True).values
