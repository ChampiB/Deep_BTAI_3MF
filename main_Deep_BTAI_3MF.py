import time
from agent.inference.TemporalSliceBuilder import TemporalSliceBuilder
from agent.dnn.PolicyNetworks import Policy4x100
from env.dSpritesEnv import dSpritesEnv
from env.wrapper.dSpritesPreProcessingWrapper import dSpritesPreProcessingWrapper
from agent.Deep_BTAI_3MF import Deep_BTAI_3MF
import torch


def main():
    """
    A simple example of how to use the BTAI_3MF framework.
    :return: nothing.
    """

    # Create the environment.
    env = dSpritesEnv(granularity=8, repeat=8)
    env = dSpritesPreProcessingWrapper(env)

    # Define the parameters of the generative model.
    a = env.a()
    b = env.b()
    c = env.c()
    d = env.d(uniform=True)

    # Define the temporal slice structure.
    ts = TemporalSliceBuilder("A_0", env.n_actions) \
        .add_state("S_pos_x", d["S_pos_x"]) \
        .add_state("S_pos_y", d["S_pos_y"]) \
        .add_state("S_shape", d["S_shape"]) \
        .add_state("S_scale", d["S_scale"]) \
        .add_state("S_orientation", d["S_orientation"]) \
        .add_observation("O_pos_x", a["O_pos_x"], ["S_pos_x"]) \
        .add_observation("O_pos_y", a["O_pos_y"], ["S_pos_y"]) \
        .add_observation("O_shape", a["O_shape"], ["S_shape"]) \
        .add_observation("O_scale", a["O_scale"], ["S_scale"]) \
        .add_observation("O_orientation", a["O_orientation"], ["S_orientation"]) \
        .add_transition("S_pos_x", b["S_pos_x"], ["S_pos_x", "A_0"]) \
        .add_transition("S_pos_y", b["S_pos_y"], ["S_pos_y", "A_0"]) \
        .add_transition("S_shape", b["S_shape"], ["S_shape"]) \
        .add_transition("S_scale", b["S_scale"], ["S_scale"]) \
        .add_transition("S_orientation", b["S_orientation"], ["S_orientation"]) \
        .add_preference(["O_pos_x", "O_pos_y", "O_shape"], c["O_shape_pos_x_y"]) \
        .build()

    # Create the policy network.
    policy = Policy4x100(env.n_states(), env.n_actions)

    # Create the agent.
    agent = Deep_BTAI_3MF(policy, ts, max_planning_steps=150, exp_const=2, lr=0.001)

    # Train the agent.
    run_trials(agent, env, n_trials=1000)

    # Test the agent.
    run_trials(agent, env, n_trials=1000, use_mcts=False)


def run_trials(agent, env, n_trials=10000, use_mcts=True, threshold=0):
    """
    Run a number of trials.
    :param agent: the agent that chosing the actions.
    :param env: the environment in which the agent is run.
    :param n_trials: the number of trial to run.
    :param use_mcts: True, if MCTS should be used, False if the policy should be used.
    :param threshold: the reward threshold above which a run is consired a success.
    :return: nothing.
    """
    # Implement the action-perception cycles.
    score = 0
    ex_times_s = torch.zeros([n_trials])
    for i in range(n_trials):
        obs = env.reset()
        env.render()
        agent.reset(obs)
        ex_times_s[i] = time.time()
        while not env.done():
            action = agent.step(use_mcts)
            obs = env.execute(action)
            env.render()
            solved = env.get_reward() > threshold
            agent.update(action, obs, solved)
        ex_times_s[i] = time.time() - ex_times_s[i]
        score += env.get_reward()

    # Display the performance of the agent.
    print("Percentage of task solved: {}".format((score + n_trials) / (2 * n_trials)))
    print("Execution time (sec): {} +/- {}".format(ex_times_s.mean().item(), ex_times_s.std(dim=0).item()))


if __name__ == '__main__':
    main()
