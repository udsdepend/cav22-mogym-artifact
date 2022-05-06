import sys

sys.path.append("../")
import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
import dataclasses as d
import typing as t
import pathlib
import random
from momba import jani, gym
from rlmate.argument_parser import Argument_parser
from dqn import Agent
import numpy as np
import random
import torch

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)

# load the formal model
network = jani.load_model(
    (
        pathlib.Path(__file__).parent
        / ".."
        / "jani-models"
        / "mdp"
        / "cdrive"
        / "cdrive.2.jani"
    ).read_text("utf8")
)

(instance,) = network.instances

if __name__ == "__main__":
    # parse the parameters for learning
    argument_parser = Argument_parser()
    dqn_arguments = argument_parser.parse() 
    # crate the env
    env = gym.create_generic_env(network, instance, "goal")
    # create and train the agent with the already parsed arguments
    agent = Agent(env, dqn_arguments)
    agent.train()

