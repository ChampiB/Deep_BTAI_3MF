from torch.nn.functional import softmax
from agent.planning.PMCTS import PMCTS
from agent.planning.MCTS import MCTS
import torch
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from agent.memory.ReplayBuffer import ReplayBuffer, Experience


class Deep_BTAI_3MF:
    """
    The class implementing the Branching Time Active Inference algorithm with
    Multi-Modalities and Multi-Factors.
    """

    def __init__(
            self, policy, ts, max_planning_steps, exp_const, lr,
            mcts_type="pmcts", queue_capacity=10000, batch_size=32
    ):
        """
        Construct the BTAI_3MF agent.
        :param policy: the policy network used to predict the output of the MCTS.
        :param ts: the temporal slice to be used by the agent.
        :param max_planning_steps: the maximum number of planning iterations.
        :param exp_const: the exploration constant of the Monte-Carlo tree search algorithm.
        :param lr: the learning rate of the policy network.
        :param queue_capacity: the size of the replay buffer.
        :param batch_size: the size of the batch to use to train the policy network.
        """
        self.policy = policy
        self.ts = ts
        self.mcts_type = mcts_type
        self.mcts = PMCTS(exp_const, self) if mcts_type == "pmcts" else MCTS(exp_const)
        self.max_planning_steps = max_planning_steps
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
        if self.mcts_type == "pmcts":
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
