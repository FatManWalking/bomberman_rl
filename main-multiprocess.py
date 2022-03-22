import os
from argparse import ArgumentParser
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld
from fallbacks import pygame

import threading
import multiprocessing
from multiprocessing.managers import BaseManager

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)

def world_controller(world, n_rounds):

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:

            world.do_step(user_input)

def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=1, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    # Interaction
    for sub in [play_parser]:
    
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

    args = parser.parse_args(argv)

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)

    else:
        raise ValueError(f"Unknown command {args.command_name}")

    world_controller(world, args.n_rounds)
    world.end()

class Data:
    def __init__(self,data=None):
        self.data = data

    def get(self):
        return self.data

    def set(self,data):
        self.data = data

def test(name,data,lock):
    lock.acquire()
    print("in thread {} name is {}".format(threading.current_thread(),name))
    print("data is {} id(data) is {}".format(data.get(),id(data)))
    data.set(data.get()+1)
    lock.release()

BaseManager.register("mydata",Data)

def getManager():
    m = BaseManager()
    m.start()
    return m

if __name__ == '__main__':
    main()