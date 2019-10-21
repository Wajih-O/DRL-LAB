import numpy as np
import random
import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from q_net import NNFactory
from replay_buffer import Experience, BaseReplayBufferFactory
from experience_replay_agent import ExperienceReplayAgent

LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def to_tensors(experiences: List[Experience], device) -> Tuple[torch.Tensor]:
    """ Transforms a batch of experiences to tensors for  training/learning
    TODO: move to_tensors this to a utils/tools deps.
    """
    states = torch.from_numpy(
        np.array([e.state for e in experiences])).float().to(device)
    actions = torch.from_numpy(
        np.vstack([e.action for e in experiences])).float().to(device)
    rewards = torch.from_numpy(
        np.vstack([e.reward for e in experiences])).float().to(device)
    next_states = torch.from_numpy(
        np.array([e.next_state for e in experiences])).float().to(device)
    dones = torch.from_numpy(
        np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
    return states, actions, rewards, next_states, dones


class LocalTarget:
    """ A local-target-optimizer net config"""
    def __init__(self, local, target, lr):
        self.local = local
        self.target = target
        self._set_optimizer(lr)

    def _set_optimizer(self, lr):
        # TODO Generalize optimizer for the local target config
        self.optimizer = optim.Adam(self.local.parameters(), lr=lr)


class Actor(LocalTarget):
    def __init__(self, nn_factory, lr, seed:int=0):
        self.local = nn_factory.build(DEVICE, seed)
        self.target = nn_factory.build(DEVICE, seed)
        self._set_optimizer(lr)
        # Noise process
        self.noise = OUNoise(self.local.action_size(), seed)

    def reset_noise(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def update(self, states, critic):
        actor_loss = -critic.local(states, self.local(states)).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

    def update_target(self, tau):
        DDPGAgent.soft_update(self.local, self.target, tau)



class DDPGAgent(ExperienceReplayAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, actor_factory: NNFactory, critic_factory: NNFactory,
                 replay_buffer_factory: BaseReplayBufferFactory, actors_nbr=1,
                 seed: int = 0, batch_size: int = 128,
                 step_to_update: int = 5, buffer_size: int = int(1e5), gamma: float = .999,
                 tau: float = 1e-3, episodes_window_size: int = 100, name=None,
                 lr_critic=LR_CRITIC, lr_actor=LR_ACTOR):
        """Initialize an Agent object.

        :param seed: initializes pseudo random gen. random.seed(seed)
        :param batch_size: training batch_size
        :param step_to_update: after which the local network is trained/and the target is updated
        :param buffer_size:  replay buffer size
        :param gamma: reward discount factor
        :param lr: learning rate
        :param tau: Q soft update tau
        :param episodes_window_size: deque storing the [episodes_window_size] last episodes score
        """
        super(DDPGAgent, self).__init__(replay_buffer_factory=replay_buffer_factory, seed=seed,
                                        batch_size=batch_size,
                                        step_to_update=step_to_update, buffer_size=buffer_size,
                                        gamma=gamma,
                                        episodes_window_size=episodes_window_size, name=name)
        self.batch_size = batch_size  # initialize learning batch size
        self.tau = tau
        self.t_step = 0

        # Actor Network (w/ Target Network)
        # self.actors = [
        #     LocalTarget(actor_factory.build(DEVICE), actor_factory.build(DEVICE), LR_ACTOR)
        #     for _ in range(actors_nbr)]
        self.actors = [ Actor(actor_factory, lr_actor, seed) for seed in range(actors_nbr)]
        # Critic Network (w/ Target Network)
        self.critic = LocalTarget(critic_factory.build(DEVICE), critic_factory.build(DEVICE),
                                  lr_critic)

        # Noise process
        # self.noise = OUNoise(actor_factory.action_size, seed)

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from
        :param  target_model: weights will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def explore(self, env, extract_state, n_episodes=1800, max_t=1000, brain_name=None):
        """ Explore/solve the env

        :param env: the environment to solve
        :param extract_state: state extractor function (env-info->state [np.ndarray])
        :param n_episodes: the max number of exploration episodes (the env expected to be solve before)
        :param max_t: maximum number of steps/actions with in an episode (if termination is not reached)
        :param brain_name: unity env brain_name

        """

        # TODO: A small agent/env compatibility sanity check.
        # Choose the first env brain name as default if not specified
        brain_name_ = brain_name
        if brain_name_ is None:
            brain_name_ = env.brain_names[0]


        for i_episode in range(1, n_episodes + 1):
            scores = np.zeros(len(self.actors))
            # reset the environment and get initial state
            states = extract_state(env.reset(train_mode=True)[brain_name_])
            # reset noise
            for actor in self.actors:
                actor.reset_noise()
            for _ in range(max_t):
                actions = self.act(states)
                env_info = env.step(actions)[brain_name_]  # send actions to the environment

                next_states = env_info.vector_observations
                rewards = env_info.rewards  # get the reward
                dones = env_info.local_done  # see if episode has finished
                scores += np.array(rewards)  # update the score
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.step(state, action, reward, next_state, done)
                states = next_states  # roll over the state to next time step
                if done:
                    break

            self.scores_window.append(np.mean(scores))  # save most recent score
            self.scores.append(np.mean(scores))  # save most recent score

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                               np.mean(self.scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                                   np.mean(self.scores_window)))
            if np.mean(self.scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - 100, np.mean(
                        self.scores_window)))
                self.save()
                break

    def step(self, state, action, reward, next_state, done):
        """ Update the replay buffer and eventually train (each self.steps_to_update)"""
        self.memory.add(state, action, reward, next_state,
                        done, 0)  # update replay buffer with the exp.
        # Learn every self.steps_to_update steps.
        self.t_step = (self.t_step + 1) % self.steps_to_update
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # enough experience were sampled to build the batch
                self.learn()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = np.zeros((len(states), self.actors[0].local.action_size()))
        # for index, actor, state in zip (range(len(self.actors)), self.actors, states):
        #     state = torch.from_numpy(state).float().to(DEVICE)
        #     actor.local.eval()
        #     with torch.no_grad():
        #         action = actor.local(state).cpu().data.numpy()
        #     actor.local.train()
        #     if add_noise:
        #         action += self.noise.sample()
        #     actions[index,:] = action
        # return np.clip(actions, -1, 1)
        for index, state in enumerate(states):
            actions[index,:] = self.actors[index].act(state, add_noise=add_noise)
        return  actions


    def reset(self):
        self.noise.reset()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        states, actions, rewards, next_states, dones = to_tensors(self.memory.sample(), DEVICE)

        for actor in self.actors:
            # update critic
            # Get predicted next-state actions and Q values from target models
            actions_next = actor.target(
                next_states)  # predict next-state actions (with the current actor)
            # Q values from target models
            Q_targets_next = self.critic.target(next_states, actions_next)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic.local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # # clipping
            # torch.nn.utils.clip_grad_norm(self.critic.local.parameters(), 1)
            self.critic.optimizer.step()

        for actor in self.actors:
            # # update actors
            # actor_loss = -self.critic.local(states, actor.local(states)).mean()
            # # Minimize the loss
            # actor.optimizer.zero_grad()
            # actor_loss.backward()
            # actor.optimizer.step()
            actor.update(states, self.critic)

        # soft-update target networks (critic and actors)
        DDPGAgent.soft_update(self.critic.local, self.critic.target, self.tau)
        for actor in self.actors:
            actor.update_target(self.tau)


