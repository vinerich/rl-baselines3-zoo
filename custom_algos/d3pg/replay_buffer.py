import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer


class PriorityReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    indices: th.Tensor
    weights: th.Tensor


class PriorityReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments  
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        optimize_memory_usage=False
    ):
        super(PriorityReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.priorities[self.pos] = np.array(1 if self.pos == 0 and self.full is not True else self.priorities.max())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos

        # Get probabilities for sample selection
        priorities = self.priorities[:upper_bound].squeeze() * self.alpha
        probabilites = priorities / priorities.sum()

        # Select samples
        batch_inds = np.random.choice(upper_bound, batch_size, p=probabilites)

        # Get importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        weights = (upper_bound * probabilites[batch_inds]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return self._get_samples(batch_inds, weights, env)

    def _get_samples(self, batch_inds: np.ndarray, weights: np.ndarray, env: Optional[VecNormalize] = None) -> PriorityReplayBufferSamples:
        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            batch_inds,
            weights
        )
        return PriorityReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def update_priorities(self, batch_indices, batch_priorities):
        # print(batch_priorities)
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
