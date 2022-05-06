import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
import dataclasses as d
import typing as t
import pathlib
import random
from momba import jani, gym
from rlmate.argument_parser import Argument_parser
from dqn import Agent
import racetrack
from racetrack.console import format_track

@d.dataclass(frozen=True)
class Renderer:
    track: racetrack.tracks.Track

    def render(self, state: gym.abstract.StateVector, mode: str):
        print(
            format_track(
                self.track,
                racetrack.model.Coordinate(
                    state.global_env["car_x"].as_int, state.global_env["car_y"].as_int
                ),
            )
        )


# create the formal model
track = racetrack.tracks.BARTO_SMALL
scenario = racetrack.model.Scenario(
    track,
    start_cell=None,
    max_speed=50,
    fuel_model=None,
    underground=racetrack.model.Underground.SLIPPERY_TARMAC,
    compute_distances=True,
    random_start=True,
)
network = racetrack.model.construct_model(scenario)

CONTROLLED_AUTOMATON_NAME = "car"
GOAL_PROPERTY_NAME = "goalProbability"

controlled_instance = next(
    instance
    for instance in network.instances
    if instance.automaton.name == CONTROLLED_AUTOMATON_NAME
)


if __name__ == "__main__":
    from momba.engine.explore import disable_exploration_cache
    disable_exploration_cache()
    # parse the parameters for learning
    argument_parser = Argument_parser()
    dqn_arguments = argument_parser.parse()
    # create the env 
    env = gym.create_generic_env(network, controlled_instance, GOAL_PROPERTY_NAME)
    # create and train the agent with the already parsed arguments
    agent = Agent(env, dqn_arguments)
    agent.train()
