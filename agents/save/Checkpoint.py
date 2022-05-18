import os
from pathlib import Path
from singletons.Device import Device
import importlib
import torch


#
# Class allowing to load model checkpoints.
#
class Checkpoint:

    def __init__(self, config, file):
        """
        Construct the checkpoint from the checkpoint file.
        :param config: the hydra configuration.
        :param file: the checkpoint file.
        """

        # If the path is not a file, return without trying to load the checkpoint.
        if not os.path.isfile(file):
            print("[WARNING] Could not load model from: " + file)
            self.checkpoint = None
            return

        # Load checkpoint from path.
        self.checkpoint = torch.load(file, map_location=Device.get())

        # Store the configuration
        self.config = config

    def exists(self):
        """
        Check whether the checkpoint file exists.
        :return: True if the checkpoint file exists, False otherwise.
        """
        return self.checkpoint is not None

    def load_model(self, ts, training_mode=True):
        """
        Load the model from the checkpoint.
        :param ts: the temporal slice of the model.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the loaded model or None if an error occured.
        """

        # Check if the checkpoint is loadable.
        if not self.exists():
            return None

        # Load the agent class and module.
        agent_module = importlib.import_module(self.checkpoint["agent_module"])
        agent_class = getattr(agent_module, self.checkpoint["agent_class"])

        # Load the parameters of the constructor from the checkpoint.
        param = agent_class.load_constructor_parameters(self.checkpoint, ts, training_mode)

        # Instantiate the agent.
        return agent_class(**param)

    @staticmethod
    def create_dir_and_file(checkpoint_file):
        """
        Create the directory and file of the checkpoint if they do not already exist.
        :param checkpoint_file: the checkpoint file.
        :return: nothing.
        """
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            file = Path(checkpoint_file)
            file.touch(exist_ok=True)

    @staticmethod
    def set_training_mode(neural_net, training_mode):
        """
        Set the training mode of the neural network sent as parameters.
        :param neural_net: the neural network whose training mode needs to be set.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: nothing.
        """
        if training_mode:
            neural_net.train()
        else:
            neural_net.eval()

    @staticmethod
    def load_polciy(checkpoint, training_mode=True, n_states_key="n_states"):
        """
        Load the policy from the checkpoint.
        :param checkpoint: the checkpoint.
        :param n_states_key: the key of the dictionnary containing the number of states.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the critic.
        """

        # Load critic network.
        policy_module = importlib.import_module(checkpoint["policy_net_module"])
        policy_class = getattr(policy_module, checkpoint["policy_net_class"])
        policy = policy_class(
            n_states=checkpoint[n_states_key], n_actions=checkpoint["n_actions"]
        )
        policy.load_state_dict(checkpoint["policy_net_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(policy, training_mode)
        return policy

    @staticmethod
    def load_object_from_dictionary(checkpoint, key):
        """
        Load the action selection strategy from the checkpoint.
        :param checkpoint: the checkpoint.
        :param key: the key in the dictionary where the object has been serialized.
        :return: the action selection strategy.
        """

        # Load the action selection strategy from the checkpoint.
        action_selection = checkpoint[key]
        action_selection_module = importlib.import_module(action_selection["module"])
        action_selection_class = getattr(action_selection_module, action_selection["class"])
        return action_selection_class(**action_selection)
