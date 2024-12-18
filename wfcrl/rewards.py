from abc import ABC, abstractmethod

import numpy as np


class RewardShaper(ABC):
    @abstractmethod
    def __call__(self, reward: float):
        pass

    def update(self):
        pass

    def reset(self):
        pass


class DoNothingReward(RewardShaper):
    """
    Dummy class. Returns the same reward.
    """

    def __call__(self, reward):
        return reward


class ReferencePercentage(RewardShaper):
    def __init__(self, reference: float):
        self.reference = reference

    def __call__(self, reward):
        return (reward - self.reference) / self.reference


class StepPercentage(RewardShaper):
    def __init__(self, reference: float = 0.0):
        self.reference = reference

    def __call__(self, reward):
        if self.reference == 0:
            shaped_reward = 0.0
        else:
            shaped_reward = (reward - self.reference) / self.reference
        self.reference = reward
        return shaped_reward

    def reset(self, reference: float = 0.0):
        self.reference = reference


class FilteredStep(StepPercentage):
    def __init__(self, reference: float = 0.0, threshold: float = 0.0, reward_type: str = 'shaped'):
        super().__init__(reference)
        self.threshold = threshold
        self.name = "filtered_step"
        self.reward_type = reward_type

    def __call__(self, reward : float = 0, timestep : int = 0, load_penalty = 0):
        shaped_reward = self.compute_reward(reward, self.reference)
        self.reference = reward
        return shaped_reward

    def compute_reward(self, reward, reference):
        shaped_reward = 0.0
        percentage = 0
        if reference != 0:
            percentage = (reward - reference) / np.abs(reference)
            if np.abs(percentage) > self.threshold:
                shaped_reward = np.min(
                    (np.max((np.floor(np.abs(percentage) / self.threshold) * np.sign(percentage), -3)), 3))

        if self.reward_type == 'shaped':  # staircase reward
            return shaped_reward
        elif self.reward_type == 'sign':  # reward based on the sign of improvement
            return np.sign(percentage)
        elif self.reward_type == 'power':  # reward based on the power of the farm
            return reward


class RewardSum(RewardShaper):
    def __init__(self, reference: float = 0.0):
        self.reference = reference
        self.name = "power_plus_change"

    def __call__(self, reward, timestep : int = 0, load_penalty = 0):
        if self.reference == 0:
            shaped_reward = 0.0
        else:
            shaped_reward = np.sign((reward - self.reference) / np.abs(self.reference))
        self.reference = reward
        return reward + shaped_reward

    def reset(self, reference: float = 0.0):
        self.reference = reference


class TrackReward():
    def __init__(self, reference, threshold: float = 0.0):
        self.reference = reference

    def __call__(self, reward :  float , timestep : int, load_penalty : float = 0):
        return self.compute_reward(reward, timestep, load_penalty)

    def reset(self, reference: float = 0.0):
        pass

    def compute_reward(self, reward :  float , timestep : int, load_penalty : float = 0):
        percentage = (self.reference[timestep] - reward) ** 2 / self.reference[timestep] ** 2

        # print(percentage)
        return 5 - percentage * 100