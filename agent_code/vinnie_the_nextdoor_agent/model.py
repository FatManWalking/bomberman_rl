import numpy as np
from .utils import state_to_features
from collections import defaultdict

# from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

# from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

# from sklearn.classifier_selection import train_test_split

# import neptune.new as neptune
# import neptune.new.integrations.sklearn as npt_utils
# import json

GAMMA = 0.95
LEARNING_RATE = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.2
EXPLORATION_DECAY = 0.96


class DQNSolver:
    def __init__(self, game, actions):
        self.exploration_rate = EXPLORATION_MAX
        self.game = game
        self.action_space = len(actions)
        self.actions = actions

        self.alpha = 0.3  #
        self.gamma = 0.7  #
        self.epsilon = 1
        self.min_exploration = 0.3
        self.exploration_decay = 0.98

        # self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
        # self.classifier = MultiOutputRegressor(
        #    LGBMRegressor(n_estimators=100, n_jobs=-1)
        # )
        self.classifier = MultiOutputRegressor(
            KNeighborsRegressor(n_jobs=-1, n_neighbors=3)
        )
        # self.classifier = MultiOutputRegressor(SVR(), n_jobs=8)
        self.isFit = False

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Don't pickle the neptune ai logger
    #     del state["game"].__dict__["run"]
    #     del state["game"].__dict__["npt_utils"]

    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Add back since it doesn't exist in the pickle
    #     with open("secret.json", "r") as f:
    #         dic = json.load(f)
    #     self.game.run = neptune.init(
    #         project="fatmanwalking/Bomberman", api_token=dic["api_token"],
    #     )
    #     self.game.npt_utils = npt_utils

    def experience_replay(self, q_table):

        table = np.zeros((len(q_table.q_table), 31))
        actions = np.zeros((len(q_table.q_table), 6))

        for (key, value), i in zip(
            q_table.q_table.items(), range(len(q_table.q_table))
        ):

            table[i] = key
            actions[i] = value

        self.classifier.fit(table, actions)

        # self.game.run["classifier"] = self.game.npt_utils.create_classifier_summary(
        #    self.classifier, table, table, actions, actions
        # )

        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


class Q_Table:
    def __init__(self, game, actions) -> None:

        self.alpha = 0.3  #
        self.gamma = 0.7  #
        self.epsilon = 1
        self.min_exploration = 0.3
        self.exploration_decay = 0.98

        self.game = game

        self.actions = actions

        self.q_table = defaultdict(
            lambda: np.zeros([game.action_space_size])
        )  # We should start small and build as goes for faster look ups and less memory usage

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Don't pickle the neptune ai logger
    #     del state["game"].__dict__["run"]
    #     del state["game"].__dict__["npt_utils"]

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Add back since it doesn't exist in the pickle
    #     with open("secret.json", "r") as f:
    #         dic = json.load(f)
    #     self.game.run = neptune.init(
    #         project="fatmanwalking/Bomberman", api_token=dic["api_token"],
    #     )
    #     self.game.npt_utils = npt_utils

    def choose_action(self, features):

        action_index = np.argmax(self.q_table[features])
        return self.actions[action_index]

    def update_q(self, batch):

        for sample in batch:

            old_ft = sample[0]
            action_index = sample[1]
            new_ft = sample[2]
            rewards = sample[3]

            if type(old_ft) is not tuple:
                self.game.logger.debug(f"{old_ft} was not a tuple")
                return None

            if type(new_ft) is not tuple:
                self.game.logger.debug(f"{new_ft} was not a tuple")
                return None

            # We do not have to check if either of those exist
            # If they dont default dict creates an np.zero array for us with that key
            exspected_reward = self.q_table[old_ft][action_index]
            max_next_reward = np.max(self.q_table[new_ft])

            # The actual Q update step based on temporal difference
            updated_q = (1 - self.alpha) * exspected_reward + self.alpha * (
                rewards + self.gamma * max_next_reward
            )
            self.q_table[old_ft][action_index] = updated_q

            if self.epsilon < 1 - self.min_exploration:
                self.epsilon *= self.exploration_decay

    def update_terminal(self, old_game_state, self_action, rewards):

        action_index = self.actions[self_action]
        old_ft = state_to_features(old_game_state)

        if type(old_ft) is not tuple:
            self.game.logger.debug(f"{old_ft} was not a tuple")
            return None

        # We do not have to check if either of those exist
        # If they dont default dict creates an np.zero array for us with that key
        exspected_reward = self.q_table[old_ft][action_index]

        # The actual Q update step based on temporal difference
        updated_q = (1 - self.alpha) * exspected_reward + self.alpha * rewards

        self.q_table[old_ft][action_index] = updated_q

