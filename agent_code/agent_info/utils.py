import numpy as np
from collections import defaultdict
from random import shuffle
from typing import Dict, List, Tuple

# So we do not have to maintain this in multiple locations
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    vision = list(awarness(game_state))
    target = closest_target(game_state)

    if not target == None:
        vision.extend(list(target))

    return tuple(vision)


def closest_target(game_state):
    _, _, _, start = game_state["self"]

    free_space = game_state["field"] == 0

    targets = game_state["coins"]

    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def relative_position(items: List[Tuple], agent: Tuple) -> List[Tuple]:
    """
    Takes the original coordinates and recalculates them relative to the agent position as 0,0
    This limits possible states without losing information
    """
    # TODO: Clean this up, the slight alteration item[0] is needed because bomb tuple is ((X,Y),Turns) and Coins just (X,Y)
    try:
        relative_position = [
            tuple(map(lambda i, j: i - j, item[0], agent)) for item in items
        ]
    except:
        relative_position = [
            tuple(map(lambda i, j: i - j, item, agent)) for item in items
        ]
    return relative_position


def rotation(game_state: Dict):

    """
    #IDEA not used for now

    Rotates all actions as if the agent always starts on the top left.
    This lets every start always look similar and should result in more stable starting strategy
    """
    global ACTIONS

    if game_state["self"][3] == (1, 1):
        pass
    if game_state["self"][3] == (15, 1):
        pass
    if game_state["self"][3] == (1, 15):
        pass
    if game_state["self"][3] == (15, 15):
        pass


def vision_field(game_state: Dict):

    # Position of the agent
    self_pos = game_state["self"][3]

    # How far can you look
    vision = 1

    # Game Field at the position of the agent
    left = self_pos[0] - vision
    right = self_pos[0] + vision + 1

    down = self_pos[0] - vision
    top = self_pos[0] + vision + 1

    return game_state["field"][left:right, down:top]


def awarness(game_state: Dict):
    """
    With this their is no need to pass the 
    """
    vis_field = vision_field(game_state)

    return vis_field.flatten()


def danger(game_state: Dict):

    # implement danger posed by bombs that are about to set off
    ...

