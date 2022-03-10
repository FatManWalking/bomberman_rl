"""
Feld 15x15 field
0 = free, 1 = crate, 2 = Wall, 3 = Enemy, 4 = Bomb = 5 fieldStates
Explosion auf Feld oder nicht also 0,1 = 2 (Da Coins z.B. bei Explosion erhalten bleiben) explosionStates
aktive Bomben von Gegner 0,1,2 A | 0,1,2 B | 0,1,2 C | 0,1,2 D = 12 (Gefahr durch Gegner oder nicht) was mit Bombe im nächsten Zug? activeBombStates
Bombe im Radius 0,1 = 2 radiusBombStates
collecting coins necessary? 0,1 = 2 Kann man gerade gewinnen, wenn man sich nur auf das Töten des Gegners fokussiert? Münzen kann man ignorieren bis neu evaluiert
collectingCoinsState

15x15x5x2x12x2x2 = 108000 States
"""
import enum
import json
import os
import pickle
import random
import settings as settings

import numpy as np

field = 5 * 5
fieldStates = 7  # free = 0, wall + explosion = 1, crate = 2, coin = 3, bomb = 4, player = 5, dangerous = 6
activeBombsStates = 3 * 3 * 3  # no, bomb, nextRoundBomb for every player <============================ abhängig von Anzahl Gegner?
collectingCoinsState = 2

# observation_space = field * fieldStates * activeBombsStates * collectingCoinsState  # 9450

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

epsilon = 0.1

states = np.array([])


def setup(self):
    if not os.path.isfile("model.npy"):
        print("Setting up model from scratch.")
        self.model = np.zeros([1, len(ACTIONS)])
        self.states = np.zeros([1, 26])
    else:
        print("Loading model from saved state.")
        #with open("my-saved-model.json", "rb") as file:
        #     self.model = np.array(json.load(file))
        #with open("my-saved-states.json", "rb") as file:
        #     self.states = np.array(json.load(file))
        #self.model = np.loadtxt('model.csv', delimiter=',')
        #self.states = np.loadtxt('states.csv', delimiter=',')
        self.model = np.load('model.npy')
        self.states = np.load('states.npy')



def act(self, game_state: dict) -> str:

    currentState = find_state(self, game_state)

    if self.train:
        if random.uniform(0, 1) < epsilon:
            return random.choice(ACTIONS)  # Explore action space
        else:
            if max(self.model[currentState]) == 0:
                return random.choice(ACTIONS)
            return find_action_by_index(np.argmax(self.model[currentState]))  # Exploit learned values
    else:
        if max(self.model[currentState]) == 0:
            return random.choice(ACTIONS)
        return find_action_by_index(np.argmax(self.model[currentState]))


# comparing with existing states and if state exist, return index, if not, save and return new index
def find_state(self, game_state: dict) -> int:
    _, score, bombs_left, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]
    playerHaveBombs = [b for (n, s, b, xy) in game_state['others']]
    bombs = [xy for (xy, t) in game_state['bombs']]
    coins = game_state['coins']
    arena = game_state['field']  # maybe in setup()

    fov = create_fov(x, y)

    #dangerousFields = create_dangerous_fields(bombs)

    checkState = []

    # get the States from the fields in fov
    for coord in fov:
        fieldState = int
        if coord in arena:  # order is important
            if arena[coord] == 0:
                fieldState = States.free
            if coord in coins:
                fieldState = States.coin
            if arena[coord] == 1:
                fieldState = States.crate
            if coord in others:
                fieldState = States.player
            #if coord in dangerousFields:
            #    fieldState = States.dangerous
            if arena[coord] == -1:
                fieldState = States.wall

            if coord in bombs:
                fieldState = States.bomb
        else:
            fieldState = States.wall  # outside of the arena

        checkState = np.append(checkState, fieldState)

    # find all active bombs Todo was wenn Spieler stirbt? Kp wer stirbt
    # for ready in playerHaveBombs:
    #    if ready:
    #       checkState = np.append(checkState, States.hasBomb)
    #    else:
    #        checkState = np.append(checkState, States.noBomb)
    #    # Todo logic: States.nextRoundBomb
    #   # checkState = checkState.append(States.nextRoundBomb)
    #   checkState = np.append(checkState, States.hasBomb)

    # check if collecting coins is necessary
    # Todo logic: if collecting coins is necessary
    checkState = np.append(checkState, States.collect)

    for index, s in enumerate(self.states):
        if np.array_equal(s, checkState):
            return index

    self.states = np.append(self.states, [checkState], axis=0)
    self.model = np.append(self.model, [[0, 0, 0, 0, 0, 0]], axis=0)

    for index, s in enumerate(self.states):
        if np.array_equal(s, checkState):
            np.save('states.npy', states)
            return index

    return -1  # fail


def find_index_from_action(action: str) -> int:
    # no switch-case :(

    if action == 'UP':
        return 0
    if action == 'RIGHT':
        return 1
    if action == 'DOWN':
        return 2
    if action == 'LEFT':
        return 3
    if action == 'WAIT':
        return 4
    if action == 'BOMB':
        return 5


def find_action_by_index(index: int) -> str:
    # no switch-case :(

    if index == 0:
        return 'UP'
    if index == 1:
        return 'RIGHT'
    if index == 2:
        return 'DOWN'
    if index == 3:
        return 'LEFT'
    if index == 4:
        return 'WAIT'
    if index == 5:
        return 'BOMB'
    return -1


# coords of fields in fov
def create_fov(x, y) -> []:
    addX = - 2  # Todo logic: depends on fov... so this is not variable atm
    addY = - 2  # left upper corner

    fov = np.zeros([field, 2])
    for coord in fov:
        coord[0] = x + addX
        coord[1] = y + addY

        """
            y - 1 == 'UP'
            y + 1 == 'DOWN'
            x - 1 == 'LEFT'
            x + 1 == 'RIGHT'
        """

        addX += 1
        if addX > 2:
            addX = -2
            addY += 1
    return fov


# coords of dangerous fields
def create_dangerous_fields(bombs) -> []:
    explosionSize = settings.BOMB_POWER * settings.BOMB_POWER
    dangerousFields = []
    dangerousFieldsX = -1  # Todo logic: depends on bombSize... so this is not variable atm
    dangerousFieldsY = -1  # left upper corner
    for b in bombs:
        xb = b[0]
        yb = b[1]
        temp = []
        for i in np.arange(1, explosionSize):
            temp = np.append(xb + dangerousFieldsX)
            temp = np.append(yb + dangerousFieldsY)

            dangerousFieldsX += 1
            if dangerousFieldsX > 1:
                dangerousFieldsX = -1
                dangerousFieldsY += 1
            if dangerousFieldsY > 1:
                dangerousFieldsY = -1
            dangerousFields = np.append(dangerousFields, [temp], axis=0)
    return dangerousFields


class States(enum.Enum):
    free = 0
    wall = 1
    crate = 2
    coin = 3
    bomb = 4
    player = 5
    dangerous = 6
    noBomb = 7
    hasBomb = 8
    nextRoundBomb = 9
    collect = 10
    kill = 11
