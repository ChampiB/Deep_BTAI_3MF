import collections
import numpy as np
from torch import cat, FloatTensor, BoolTensor, IntTensor, unsqueeze
from singletons.Device import Device


#
# Class storing an experience.
#
Experience = collections.namedtuple('Experience', field_names=['state', 'mcts_result'])


#
# Class implementing the experience replay buffer.
#
class ReplayBuffer:

    def __init__(self, capacity=10000):
        """
        Constructor
        :param capacity: the number of experience the buffer can store
        """
        self.device = Device.get()
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        """
        Getter
        :return: the number of elements contained in the replay buffer
        """
        return len(self.buffer)

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        :return: nothing
        """
        self.buffer.append(experience)

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer
        :param batch_size: the size of the batch to sample
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, mcts_result = zip(*[self.buffer[idx] for idx in indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return \
            self.list_to_tensor(state).to(self.device), \
            self.list_to_tensor(mcts_result).to(self.device)
