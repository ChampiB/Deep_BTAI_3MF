import time
from agents.inference.TemporalSliceBuilder import TemporalSliceBuilder
from env.wrapper.dSpritesPreProcessingWrapper import dSpritesPreProcessingWrapper
import torch
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from agents.save.Checkpoint import Checkpoint
import hydra
import numpy as np
import random


@hydra.main(config_path="config", config_name="training")
def main(config):
    """
    A simple example of how to use the train a Deep_BTAI_3MF agent.
    :param config: the hydra configuration.
    :return: nothing.
    """

    # Set the seed requested by the user.
    set_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    print("[INFO] Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment.
    env = instantiate(config["env"])
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

    # Update the config to enable the instantiation of the agent.
    with open_dict(config):
        config.agent.policy.n_states = env.n_states()

    # Create the agent.
    archive = Checkpoint(config, config["checkpoint"]["file"])
    agent = archive.load_model(ts) if archive.exists() else instantiate(config["agent"], ts=ts)

    # Train the agent.
    # TODO run_trials(agent, env, config, n_trials=config["n_trials"])

    # Test the agent.
    run_trials(agent, env, config, n_trials=100, use_mcts=False)


def set_seed(seed):
    """
    Set the seed requested by the configuration.
    :param seed: the seed to be set.
    :return: nothing.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def run_trials(agent, env, config, n_trials=10000, use_mcts=True, threshold=0):
    """
    Run a number of trials.
    :param agent: the agent that chosing the actions.
    :param env: the environment in which the agent is run.
    :param config: the hydra configuration.
    :param n_trials: the number of trial to run.
    :param use_mcts: True, if MCTS should be used, False if the policy should be used.
    :param threshold: the reward threshold above which a run is consired a success.
    :return: nothing.
    """
    # Implement the action-perception cycles.
    score = 0
    ex_times_s = torch.zeros([n_trials])
    for i in range(n_trials):
        # Reset the agent and environment.
        obs = env.reset()
        env.render(config["display_gui"])
        agent.reset(obs)
        ex_times_s[i] = time.time()

        # Implement the action-perception cycle.
        while not env.done():
            action = agent.step(use_mcts)
            obs = env.execute(action)
            env.render(config["display_gui"])
            solved = env.get_reward() > threshold
            agent.update(action, obs, solved)

        # Keep track of performance and execution time.
        ex_times_s[i] = time.time() - ex_times_s[i]
        score += env.get_reward()

        # Save the agent, if needed.
        if (i + 1) % config["checkpoint"]["frequency"] == 0:
            agent.save(config)

    # Display the performance of the agent.
    print("[INFO] Percentage of task solved: {}".format((score + n_trials) / (2 * n_trials)))
    print("[INFO] Execution time (sec): {} +/- {}".format(ex_times_s.mean().item(), ex_times_s.std(dim=0).item()))


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    main()
