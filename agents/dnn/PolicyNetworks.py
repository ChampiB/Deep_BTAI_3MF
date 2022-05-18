import torch
import torch.nn as nn
from torch import cat
from singletons.Device import Device


class Policy4x100(nn.Module):
    """
    A class implementing a policy network composed of four fully
    connected layers of size 100.
    """

    def __init__(self, n_states, n_actions):
        """
        Create a policy network composed of four fully connected
        layers of size 100.
        :param n_states: a list containing the number of states for each latent factor.
        :param n_actions: the number of actions in the environment.
        """

        super().__init__()

        # Store the number of states for each factor.
        self.n_states = n_states

        # Create the transition network.

        n_states = sum(n_state for _, n_state in n_states.items())
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def pre_processing(states):
        """
        Pre-process the states passed as parameter.
        :param states: the state to pre-process.
        :return: nothing.
        """
        states = [state for _, state in states.items()]
        if states[0].dim() == 1:
            states = [torch.unsqueeze(state, dim=0) for state in states]
        return cat(states, dim=1).to(torch.float).to(Device.get())

    def forward(self, states):
        """
        Forward pass through the transition network.
        :param states: the input states.
        :return: the distribution over action.
        """
        x = self.pre_processing(states) if isinstance(states, dict) else states
        out = self.__net(x)
        return out, torch.log(out + 1e-20)
