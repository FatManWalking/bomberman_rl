import numpy as np

import events as e
from .callbacks import find_state, find_index_from_action

"""

"""


def setup_training(self):
    """

    """


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    experience
    is called once after each step except the last. At this point, all the actions have been executed
    and their consequences are known. Use this callback to collect training data and fill an
    experience buffer.
    """
    alpha = 0.1
    gamma = 0.6
    if old_game_state is None:
        old_state = find_state(self, new_game_state)
    else:
        old_state = find_state(self, old_game_state)
    new_state = find_state(self, new_game_state)

    old_value = self.model[old_state, find_index_from_action(self_action)]
    next_max = np.max(self.model[new_state])

    new_value = (1 - alpha) * old_value + alpha * (calculate_rewards(events) + gamma * next_max)
    self.model[old_state, find_index_from_action(self_action)] = new_value


def end_of_round(self, last_game_state, last_action, events):
    # Store the model
    #with open("my-saved-model.json", "wb") as file:
    #    json.dump(self.model.toList(), file) #Todo save states
    #with open("my-saved-states.json", "wb") as file:
    #    json.dump(states.toList(), file)
    np.savetxt('model.csv', self.model, delimiter=',')
    #np.savetxt('states.csv', states, delimiter=',')
    np.save('model.npy', self.model)


def calculate_rewards(events) -> int:

    # if state kill enemy and dont collect, change rewards

    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 1,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 15,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -15,
        e.SURVIVED_ROUND: 20
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum
