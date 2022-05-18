from torch.nn.functional import softmax
from agents.planning.PMCTS import PMCTS
from agents.planning.MCTS import MCTS
import torch
from torch.optim import Adam
from singletons.Device import Device
from agents.save.Checkpoint import Checkpoint
from torch.distributions.categorical import Categorical
from agents.memory.ReplayBuffer import ReplayBuffer, Experience


class Deep_BTAI_3MF:
    """
    The class implementing the Branching Time Active Inference algorithm with
    Multi-Modalities and Multi-Factors.
    """

    def __init__(
            self, policy, mcts_type, ts, max_planning_steps, exp_const,
            lr, queue_capacity=10000, batch_size=32, **_
    ):
        """
        Construct the BTAI_3MF agent.
        :param policy: the policy network used to predict the output of the MCTS.
        :param planning: the planning algorithm to be used.
        :param ts: the temporal slice to be used by the agent.
        :param max_planning_steps: the maximum number of planning iterations.
        :param exp_const: the exploration constant of the Monte-Carlo tree search algorithm.
        :param lr: the learning rate of the policy network.
        :param queue_capacity: the size of the replay buffer.
        :param batch_size: the size of the batch to use to train the policy network.
        """
        # Model related attributes.
        self.policy = policy
        Device.send([self.policy])
        self.ts = ts

        # Planning related attributes.
        self.exp_const = exp_const
        self.mcts_type = mcts_type
        self.mcts = PMCTS(exp_const, self) if mcts_type == "pmcts" else MCTS(exp_const)
        self.max_planning_steps = max_planning_steps

        # Learning related attributes.
        self.queue_capacity = queue_capacity
        self.lr = lr
        self.policy_optimiser = Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.experiences = []
        self.batch_size = batch_size

    def reset(self, obs):
        """
        Reset the agent to its pre-planning state.
        :param obs: the observation that must be used in the computation of the posterior.
        :return: nothing.
        """
        self.ts.reset()
        self.ts.i_step(obs)

    def step(self, using_mcts=True):
        """
        Perform planning and action selection.
        :param using_mcts: True if MCTS should be performed, False if the policy should be used for action selection.
        :return: the action to execute in the environment.
        """
        # If MCTS should not be used for action selection, then use policy network instead.
        if not using_mcts:
            act_prob, _ = self.policy(self.ts.states_posterior)
            return Categorical(act_prob).sample().item()

        # Perform MCTS.
        for i in range(0, self.max_planning_steps):
            node = self.mcts.select_node(self.ts)
            e_nodes = self.mcts.expansion(node)
            self.mcts.evaluation(e_nodes)
            self.mcts.propagation(e_nodes)
        best_action = max(self.ts.children, key=lambda x: x.visits).action

        # Train the policy network.
        if isinstance(self.mcts, PMCTS):
            self.train_policy()
            state = self.policy.pre_processing(self.ts.states_posterior)
            mcts_result = self.get_policy_target(best_action)
            self.experiences.append(Experience(state, mcts_result))

        # Return action to perform in the environment.
        return best_action

    def get_policy_target(self, best_action, target_type="best", normalisation_type="sum"):
        """
        Getter, returning the target of the policy network.
        :param best_action: the best action to predict if target_type == "best".
        :param target_type: the type of target to use, i.e. "best" or "avg_efe" or visits.
        :param normalisation_type: the type of normalisation to used for the target distribution.
        :return: nothing.
        """
        # Compute the target.
        target = torch.zeros(len(self.ts.children))
        for child in self.ts.children:
            if target_type == "best":
                target[child.action] = 1 if best_action == child.action else 0
            elif target_type == "avg_efe":
                target[child.action] = child.cost / child.visits
            else:
                target[child.action] = child.visits

        # Normalise the target.
        if normalisation_type == "sum":
            target /= target.sum()
        else:
            target = softmax(target, dim=0)
        return target

    def train_policy(self):
        """
        Train the policy network to predict the best action according to the MCTS.
        :return: nothing.
        """
        # Check that there is enough element in the queue.
        if len(self.buffer) < self.batch_size:
            return

        # Retrieve the input/output pair.
        states, target = self.buffer.sample(self.batch_size)

        # Compute the policy prediction.
        act_prob, log_act_prob = self.policy(states)

        # Compute the target distribution and its logarithm.
        log_target = torch.log(target + 1e-20)

        # Compute KL-divergence between prediction and target.
        kl = act_prob * (log_act_prob - log_target)
        kl = kl.sum()

        # Perform one step of gradient descent.
        self.policy_optimiser.zero_grad()
        kl.backward()
        self.policy_optimiser.step()

    def update(self, action, obs, solved):
        """
        Update the agent so that: (1) the root corresponds to the temporal slice reached
        when performing the action passed as parameters, (2) the posterior over hidden
        states takes into account the observation passed as parameters.
        :param action: the action that was executed in the environment.
        :param obs: the observation that was made.
        :param solved: True, if the task was solved, False otherwise.
        :return: nothing.
        """
        try:
            # If the child corresponding to the action selected has been expanded, retrieve it.
            self.ts = next(filter(lambda x: x.action == action, self.ts.children))
        except StopIteration:
            # Else create the child corresponding to the action selected.
            self.ts = self.ts.p_step(action)

        # Make the selected child, ready to be the new root in the next cycle.
        self.ts.reset()
        self.ts.use_posteriors_as_empirical_priors()
        self.ts.i_step(obs)

        # Update the replay buffer if the task was solved.
        if solved:
            for experience in self.experiences:
                self.buffer.append(experience)
        self.experiences.clear()

    def save(self, config):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration
        :return: nothing
        """

        # Create directories and files if they do not exist.
        checkpoint_file = config["checkpoint"]["file"]
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "n_states": config["agent"]["policy"]["n_states"],
            "n_actions": config["env"]["n_actions"],
            "batch_size": self.batch_size,
            "queue_capacity": self.queue_capacity,
            "lr": self.lr,
            "exp_const": self.exp_const,
            "mcts_type": self.mcts_type,
            "max_planning_steps": self.max_planning_steps,
            "policy_net_state_dict": self.policy.state_dict(),
            "policy_net_module": str(self.policy.__module__),
            "policy_net_class": str(self.policy.__class__.__name__),
        }, checkpoint_file)

    @staticmethod
    def load_constructor_parameters(checkpoint, ts, training_mode=True):
        """
        Load the constructor parameters from a checkpoint.
        :param checkpoint: the chechpoint from which to load the parameters.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :param ts: the temporal slice of the model.
        :return: a dictionary containing the contrutor's parameters.
        """
        return {
            "policy": Checkpoint.load_polciy(checkpoint, training_mode),
            "ts": ts,
            "lr": checkpoint["lr"],
            "batch_size": checkpoint["batch_size"],
            "queue_capacity": checkpoint["queue_capacity"],
            "exp_const": checkpoint["exp_const"],
            "mcts_type": checkpoint["mcts_type"],
            "max_planning_steps": checkpoint["max_planning_steps"],
        }
