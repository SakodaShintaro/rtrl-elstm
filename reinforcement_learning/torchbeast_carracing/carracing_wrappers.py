# CarRacing-v3 environment, wrapped to match the CarRacing setup of
# vla_streaming_rl (action repeat 4, 1000 decisions per episode, off-track
# penalty fix, early stop on a long run of negative rewards), except that
# the action space is the discrete one (continuous=False) used by IMPALA.

import gymnasium as gym
import numpy as np

REPEAT = 4


class CarRacingRewardFixWrapper(gym.Wrapper):
    """Fix CarRacing's -100 penalty for going off-track."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward < -30:
            reward += 100
        return obs, reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class AverageRewardEarlyStopWrapper(gym.Wrapper):
    """Truncate the episode once the last `window_size` rewards are all negative."""

    def __init__(self, env, window_size):
        super().__init__(env)
        self.window_size = window_size
        self.recent_rewards = []

    def reset(self, **kwargs):
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.recent_rewards = (self.recent_rewards + [reward])[-self.window_size:]
        all_negative = sum(r < 0.0 for r in self.recent_rewards) == self.window_size
        return obs, reward, terminated, truncated or all_negative, info


class TransposeObs(gym.ObservationWrapper):
    """(H, W, C) uint8 -> (C, H, W) uint8."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (c, h, w), np.uint8)

    def observation(self, obs):
        return np.ascontiguousarray(obs.transpose(2, 0, 1))


def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    env = env.env  # Unwrap the original TimeLimit wrapper (counted in frames)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
    env = CarRacingRewardFixWrapper(env)
    env = ActionRepeatWrapper(env, REPEAT)
    env = AverageRewardEarlyStopWrapper(env, 20)
    env = TransposeObs(env)
    return env


def make_vector_env(num_envs):
    # SAME_STEP: the observation returned with done=True is already the first
    # frame of the next episode, which is the Torchbeast convention.
    return gym.vector.AsyncVectorEnv(
        [make_env for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
