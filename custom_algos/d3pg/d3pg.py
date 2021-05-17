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
from stable_baselines3.common.noise import ActionNoise, OrnsteinUhlenbeckActionNoise
from custom_algos.d3pg.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from custom_algos.d3pg.policies import MlpPolicy


class D3PG(OffPolicyAlgorithm):
    """
    Distributional Deterministic Deep Policy Gradient (D3PG)

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
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 4,
        weight_decay: float = 0,
        action_dist_samples: int = 32,
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

        super(D3PG, self).__init__(
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
        self.weight_decay = weight_decay
        self.action_dist_samples = action_dist_samples

        if seed is not None:
            np.random.seed(seed)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(D3PG, self)._setup_model()
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

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            batch_size = replay_data.observations.size(0)

            # Critic update
            with th.no_grad():
                target_next_actions = self.actor_target.forward(replay_data.next_observations)
                target_next_actions_q, _ = self.critic_target.forward(
                    replay_data.next_observations, target_next_actions, self.action_dist_samples)
                target_next_actions_q = target_next_actions_q.transpose(1, 2)
                target_expected_Q = replay_data.rewards.unsqueeze(-1) + \
                    (1 - replay_data.dones.unsqueeze(-1)) * self.gamma * target_next_actions_q

            expected_Q, taus = self.critic.forward(replay_data.observations, replay_data.actions, self.action_dist_samples)

            # Quantile Huber loss
            td_error = target_expected_Q - expected_Q
            huber_1 = calculate_huber_loss(td_error, 1.0)
            quantil_1 = abs(taus - (td_error.detach() < 0).float()) * huber_1 / 1.0

            critic_loss = (quantil_1.sum(dim=1).mean(dim=1, keepdim=True) * replay_data.weights).mean()

            # Optimize critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), 1)
            self.critic.optimizer.step()

            # Actor update
            actions = self.actor.forward(replay_data.observations)
            actions_q = self.critic.get_qvalues(replay_data.observations, actions)
            actor_loss = -actions_q.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            actor_losses.append(actor_loss.item())

            self.replay_buffer.update_priorities(replay_data.indices, np.clip(
                abs(td_error.sum(dim=1).mean(dim=1, keepdim=True).data.cpu().numpy()), -1, 1))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
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

        return super(D3PG, self).learn(
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
        return super(D3PG, self)._excluded_save_params() + ["actor", "actor_target", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = []
        # saved_pytorch_variables = ["log_ent_coef"]
        return state_dicts, saved_pytorch_variables


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = th.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss


# def get_min_qValues(critic_pairs):
#     critics = [pair[0] for pair in critic_pairs]
#     return th.min(th.cat(critics, dim=1), dim=1, keepdim=True).values


def get_huber_loss(target_q, q, taus, weights):
    td_error = target_q - q
    huber_1 = calculate_huber_loss(td_error, 1.0)
    quantil_1 = abs(taus - (td_error.detach() < 0).float()) * huber_1 / 1.0

    return (quantil_1.sum(dim=1).mean(dim=1, keepdim=True) * weights).mean()
