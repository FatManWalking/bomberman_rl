from collections import namedtuple, deque
from random import sample
import numpy as np
from typing import List
from agent_code.rule_based_agent.callbacks import act as rb_act, setup as rb_setup
import events as e
from .model import Q_Table
from .utils import state_to_features, ACTIONS
import random
import dill as pickle

# from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 30  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # save-frequence , not used yet just saving at the end of each round
    self.save = 100

    # The 'model' in whatever form (NN, QT, MCT ...)
    self.model = Q_Table(self, ACTIONS)

    self.batch_size = 10

    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)
    rb_setup(self)


def train_act(self, gamestate):

    features = state_to_features(gamestate)
    if random.uniform(0, 1) > self.model.epsilon:
        # self.action is the unique action chosen by the agent
        action = rb_act(self, gamestate)
    else:
        action = self.model.choose_action(features)

    # self.logger.debug(f"Action taken:", action)

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
    # self.logger.debug(
    #    f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    # )

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    self.transitions.append(
        Transition(
            state_to_features(old_game_state),
            np.where(ACTIONS == self_action)[0][0],
            state_to_features(new_game_state),
            reward_from_events(self, events),
        )
    )

    if len(self.transitions) > self.batch_size:
        batch = sample(self.transitions, self.batch_size)
        self.model.update_q(batch)


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
    # self.logger.debug(
    #    f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    # )
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    if not self.save:
        with open("model.pt", "wb") as file:
            pickle.dump(self.model, file)
        self.save -= 1
    else:
        self.save = 100


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.MOVED_LEFT: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_DOWN: 0.1,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
