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

# create the formal model
network = jani.load_model(
    (
        pathlib.Path(__file__).parent
        / ".."
        / "jani-models"
        / "mdp"
        / "elevators"
        / "elevators.a-3-3.jani"
    ).read_text("utf8")
)

(instance,) = network.instances

if __name__ == "__main__":
    # parse the parameters for learning
    argument_parser = Argument_parser()
    dqn_arguments = argument_parser.parse() 
    # create the env
    env = gym.create_generic_env(network, instance, "goal")
    # create and train the agent with the already parsed arguments
    agent = Agent(env, dqn_arguments)
    agent.train()
   