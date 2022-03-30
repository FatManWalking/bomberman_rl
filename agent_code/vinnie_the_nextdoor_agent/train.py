from collections import namedtuple, deque
from random import sample
import numpy as np
from typing import List
from agent_code.rule_based_agent.callbacks import act as rb_act, setup as rb_setup
import events as e
from .model import DQNSolver, Q_Table
from .utils import state_to_features, ACTIONS
import random
import dill as pickle
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

# from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
BATCH_SIZE = 30
LAST_POSITION_HISTORY_SIZE = 2
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
REPETITION_EVENT = "REPETITION"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.lastPositions = deque(maxlen=LAST_POSITION_HISTORY_SIZE)

    # save-frequence , not used yet just saving at the end of each round
    self.saveCounter = 50_000
    self.log_counter = 0
    self.round_rewards = [0]

    # The 'model' in whatever form (NN, QT, MCT ...)
    if self.continue_train:
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("q_table.pt", "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.model = DQNSolver(self, ACTIONS)
        with open("model.pt", "wb") as file:
            pickle.dump(self.model, file)

        self.q_table = Q_Table(self, ACTIONS)
        with open("q_table.pt", "wb") as file:
            pickle.dump(self.q_table, file)

    self.batch_size = BATCH_SIZE
    rb_setup(self)


def train_act(self, gamestate):

    if random.uniform(0, 1) < self.model.epsilon:
        # self.action is the unique action chosen by the agent
        action = self.model.actions[random.randint(0, 5)]

    elif random.uniform(0, 1) < self.model.epsilon:
        action = rb_act(self, gamestate)

    else:
        features = state_to_features(gamestate)
        action = self.q_table.choose_action(features)

    return action


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    # Idea: Add your own events to hand out rewards
    if new_game_state["self"][3] in self.lastPositions:
        if not self_action.__eq__("BOMB" and "WAIT"):
            events.append(REPETITION_EVENT)
        if self_action.__eq__("BOMB" and "WAIT") and self.lastAction.__eq__(
            "BOMB" and "WAIT"
        ):
            events.append(REPETITION_EVENT)

    self.lastAction = self_action

    self.lastPositions.append(new_game_state["self"][3])

    self.transitions.append(
        Transition(
            state_to_features(old_game_state),
            np.where(self.q_table.actions == self_action)[0],
            state_to_features(new_game_state),
            reward_from_events(self, events),
        )
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    self.round_rewards.append([0])

    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            np.where(self.q_table.actions == last_action)[0],
            None,
            reward_from_events(self, events),
        )
    )

    if len(self.transitions) > self.batch_size:
        batch = sample(self.transitions, self.batch_size)
        self.q_table.update_q(batch)

    if self.saveCounter <= 0:
        # Store the model
        self.model.experience_replay(self.q_table)

        # Store the Q-table
        with open("q_table.pt", "wb") as file:
            pickle.dump(self.q_table, file)

        # Store the model
        with open("model.pt", "wb") as file:
            pickle.dump(self.model, file)

        self.saveCounter = 10_000

        np.save("round_rewards.npy", self.round_rewards)
        # if self.log_counter > 120_000:
        #    self.run.stop()

    self.log_counter += 1
    self.saveCounter -= 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 20,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        REPETITION_EVENT: -1,
        e.BOMB_DROPPED: 3,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 7,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -10,
        e.SURVIVED_ROUND: 20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # self.run["training/reward/step"].log(reward_sum)

    self.round_rewards[-1] += reward_sum
    return reward_sum
