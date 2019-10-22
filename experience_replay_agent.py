import random
import string
from collections import deque
import abc
from typing import List

from replay_buffer import Experience, BaseReplayBufferFactory


class BaseAgent:
    """ A Base agent"""


class ExperienceReplayAgent(object):
    """ An experience replay agent Base class"""

    def __init__(self, replay_buffer_factory: BaseReplayBufferFactory,
                 seed: int = 0, batch_size: int = 64,
                 step_to_update: int = 5, buffer_size: int = int(1e5), gamma: float = .99,
                 episodes_window_size: int = 100, name=None):
        """
        :param seed: initializes pseudo random gen. random.seed(seed)
        :param batch_size: training batch_size
        :param step_to_update: after which the local network is trained/and the target is updated
        :param buffer_size:  replay buffer size
        :param gamma: reward discount factor
        :param episodes_window_size: deque storing the [episodes_window_size] last episodes score
        """
        random.seed(seed)

        self._name = name if name is not None else ''.join(
            random.choice(string.ascii_letters) for _ in range(8))

        self.gamma = gamma  # reward discount param.

        # Experience Replay memory

        # the batch size should not be part of the buffer replay but could
        # be checked here with a warning +  sampling with replacement will be the default behavior when
        # the batch_size is bigger than the buffer size !

        self.memory = replay_buffer_factory.build(buffer_size, batch_size, seed)

        # Initialize time step (for updating every self.steps_to_update steps)
        self.t_step = 0
        self.steps_to_update = step_to_update

        # Agent history stats (collected during the env. exploration/solving)
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=episodes_window_size)

    @property
    def name(self):
        return self._name

    @property
    def buffer_size(self):
        """ gets the current reply buffer size"""
        return len(self.memory)

    @abc.abstractmethod
    def save(self) -> None:
        """ Save/Serialize the agent"""

    # @abc.abstractmethod
    # def load(self):
    #     """ Load Agent for long term training."""

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        """ Update the replay buffer and eventually train (each self.steps_to_update)"""

    @abc.abstractmethod
    def learn(self, experiences: List[Experience]):
        """
        Define Learning from experiences list (sampled from the experience buffer)
        :param experiences:
        :return:
        """
    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        """
        Define a step.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
