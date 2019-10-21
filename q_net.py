import abc
from typing import Tuple

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QFCNet(nn.Module):
    """Q fully-connected-Network . 2x(FC hidden layer + Relu activation) + FC"""

    def __init__(self, state_size: int, action_size: int, layers: Tuple[int], seed: int = 0):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param layers: FC layers defined by their size (neurones number)
        """
        super(QFCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers_dim = [state_size] + list(layers) + [action_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_layer, out_layer) for in_layer, out_layer in
             zip(self.layers_dim, self.layers_dim[1:])])

    def forward(self, state):
        """Forward input"""
        nn_output = state
        for layer in self.layers[:-1]:
            nn_output = F.relu(layer(nn_output))
        return self.layers[-1](nn_output)

    def action_size(self):
        return self.layers[-1].out_features


class ActorFCNet(nn.Module):
    """Actor network fully-connected-Network . 2x(FC hidden layer + Relu activation) + FC"""

    def __init__(self, state_size: int, action_size: int, layers: Tuple[int], seed: int = 0):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param layers: FC layers defined by their size (neurones number)
        """

        super(ActorFCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers_dim = [state_size] + list(layers) + [action_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_layer, out_layer) for in_layer, out_layer in
             zip(self.layers_dim, self.layers_dim[1:])])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        nn_output = state
        for layer in self.layers[:-1]:
            nn_output = F.relu(layer(nn_output))
        return F.tanh(self.layers[-1](nn_output))

    def action_size(self):
        """ Here the action size represents the dimension of a continuous output"""
        return self.layers[-1].out_features


class CriticFCNet(nn.Module):
    """ Critic Fully-Connected network """

    def __init__(self, state_size, action_size, state_rep_layers: Tuple[int] = (256,),
                 critic_layers: Tuple[int] = (256,), seed: int = 0):
        super(CriticFCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_rep_layers_dims = [state_size] + list(state_rep_layers)
        self.state_rep_layers = nn.ModuleList(
            [nn.Linear(in_layer, out_layer) for in_layer, out_layer in
             zip(state_rep_layers_dims, state_rep_layers_dims[1:])])

        critic_layers_dims = [action_size + list(state_rep_layers)[-1]] + list(critic_layers) + [1]
        self.critic_layers = nn.ModuleList(
            [nn.Linear(in_layer, out_layer) for in_layer, out_layer in
             zip(critic_layers_dims, critic_layers_dims[1:])])

        self.seed = torch.manual_seed(seed)

        # self.reset_parameters()

    def reset_parameters(self):
        for layer in self.state_rep_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        for layer in self.critic_layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.critic_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # Extract a (learnable) representation of the status
        output = state
        for layer in self.state_rep_layers:
            output = F.leaky_relu(layer(output))
        # Relay the concat(representation(state), output) to the critic Netword
        output = torch.cat((output, action), dim=1)
        for layer in self.critic_layers:
            output = F.leaky_relu(layer(output))
        return output


class QConvNet(nn.Module):
    """ A convolutional network for Pixel input (gray-scaled/Or channel selected)"""

    def __init__(self, action_size, seed: int = 0):
        super(QConvNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Conv. layers : conv_1 layer supports gray-scaled transformed or a selected channel
        self.conv_1 = nn.Conv2d(1, 32, 9, stride=4, padding=4)
        self.conv_2 = nn.Conv2d(32, 64, 5, stride=3, padding=1)

        # Fully connected layers
        self.fc_layers_params = [(64 * 7 * 7, 256), (256, action_size)]
        self.fc1 = nn.Linear(*self.fc_layers_params[0])
        self.fc2 = nn.Linear(*self.fc_layers_params[1])

    def forward(self, state):
        """ forward ..."""
        # Applying Conv. section
        output = F.relu(self.conv_2(F.relu(self.conv_1(state))))
        # flatten the output and relay to the FC part
        output = output.view(-1, self.fc_layers_params[0][0])
        return self.fc2(F.relu(self.fc1(output)))

    def action_size(self):
        return self.fc2.out_features


class NNFactory:
    """ Abstract Network factory Q-Network, Actor, Critic"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size


    @abc.abstractmethod
    def build(self, device, seed) -> nn.Module:
        """ abstract network builder"""


class QFCNetFactory(NNFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, state_size, action_size, layers: Tuple[int] = (128, 64, 64), seed: int = 0):
        super(QFCNetFactory, self).__init__(state_size, action_size)
        self.layers = layers

    def build(self, device, seed:int = 0):
        """ build an FC based network """
        return QFCNet(self.state_size, self.action_size, self.layers, seed).to(device)


class ActorFCNetFactory(NNFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, state_size, action_size, layers: Tuple[int] = (128, 64, 64), seed: int = 0):
        super(ActorFCNetFactory, self).__init__(state_size, action_size)
        self.layers = layers

    def build(self, device, seed:int = 0):
        """ build an FC based network """
        return ActorFCNet(self.state_size, self.action_size, self.layers).to(device)


class CriticFCNetFactory(NNFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, state_size, action_size, state_rep_layers: Tuple[int] = (256,),
                 critic_layers: Tuple[int] = (256,)):
        super(CriticFCNetFactory, self).__init__(state_size, action_size)
        self.state_rep_layers = state_rep_layers
        self.critic_layers = critic_layers

    def build(self, device, seed:int = 0):
        """ build an FC based network """
        return CriticFCNet(self.state_size, self.action_size, self.state_rep_layers,
                           self.critic_layers, seed).to(device)


class QConvNetFactory(NNFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, action_size, seed: int = 0):
        self.action_size = action_size
        self.seed = seed

    def build(self, device):
        """ build an FC based network """
        return QConvNet(self.action_size, self.seed).to(device)
