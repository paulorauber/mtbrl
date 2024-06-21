import numpy as np

from mtbrl.algorithms.bamdp import Model
from mtbrl.algorithms.bamdp import Policy


class GridWorld(Model):
    def __init__(self, layout, slip):
        Model.__init__(self, layout.size + 2, 4)

        self.layout = np.array(layout)
        self.slip = slip

        self.trap_state = self.n_states - 2
        self.goal_state = self.n_states - 1

        layout_flat = self.layout.reshape(-1)

        # up, left, down, right
        actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self._p = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):
            if (s == self.trap_state) or (s == self.goal_state):
                self._p[s, :, s] = 1.0
            elif layout_flat[s] == -1:
                self._p[s, :, self.trap_state] = 1.0
            elif layout_flat[s] == 1:
                self._p[s, :, self.goal_state] = 1.0
            else:
                neighbors = []
                for a in range(4):
                    c = np.unravel_index(s, self.layout.shape)

                    next_c = c[0] + actions[a][0], c[1] + actions[a][1]
                    next_s = np.ravel_multi_index(next_c, self.layout.shape, 'clip')

                    self._p[s, a, next_s] = 1. - self.slip

                    neighbors.append(next_s)

                for neighbor in neighbors:
                    self._p[s, :, neighbor] += self.slip / self.n_actions

    def __call__(self, s, a, next_s):
        return self._p[s, a, next_s]

    def render(self, state, reward):
        if state == self.trap_state:
            img = np.array([['#']])
        elif state == self.goal_state:
            img = np.array([['$']])
        else:
            img = np.full(self.layout.shape, '.')
            img[np.where(self.layout == 1)] = '$'
            img[np.where(self.layout == -1)] = '#'
            img[np.unravel_index(state, self.layout.shape)] = '@'

        print('State:')
        print(img)
        print(f'Reward: {reward}\n')


class FixedPolicy(Policy):
    def __init__(self, string):
        actions_keys = ['w', 'a', 's', 'd']
        self.actions = [actions_keys.index(k) for k in string]

    def __getitem__(self, states):
        return self.actions[len(states) - 1]


class InteractivePolicy(Policy):
    def __init__(self):
        self.action_map = {'w': 0, 'a': 1, 's': 2, 'd': 3}

    def __getitem__(self, states):
        return self.action_map[input('\nAction: ')]


def create_gridworld(char_matrix, slip=0.):
    # {'&': 'start', '.': 'path', '#': 'trap', '$': 'goal'}
    char_matrix = np.array(char_matrix)

    layout = np.zeros(char_matrix.shape, dtype=int)
    layout[np.where(char_matrix == '#')] = -1
    layout[np.where(char_matrix == '$')] = 1

    gw = GridWorld(layout, slip)

    initial_distribution = np.zeros(char_matrix.shape)
    initial_distribution[np.where(char_matrix == '&')] = 1.0
    initial_distribution = initial_distribution / np.sum(initial_distribution)
    initial_distribution = np.concatenate([initial_distribution.reshape(-1), [0., 0.]])

    reward = np.zeros(gw.n_states)
    reward[gw.trap_state] = -1.
    reward[gw.goal_state] = 1.

    return gw, initial_distribution, reward