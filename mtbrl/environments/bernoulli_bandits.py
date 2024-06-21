import numpy as np

from mtbrl.algorithms.bamdp import Model
from mtbrl.algorithms.bamdp import Policy


class BernoulliBandits(Model):
    def __init__(self, success_probabilities):
        Model.__init__(self, 2, len(success_probabilities))
        self.success_probabilities = np.array(success_probabilities)

        self._p = np.zeros((self.n_states, self.n_actions, self.n_states))

        self.no_reward_state, self.reward_state = 0, 1
        for action in range(self.n_actions):
            self._p[:, action, self.reward_state] = self.success_probabilities[action]
            self._p[:, action, self.no_reward_state] = 1 - self.success_probabilities[action]

    def __call__(self, s, a, next_s):
        return self._p[s, a, next_s]

    def render(self, state, reward):
        print(f'Success probabilities: {dict(zip(range(1, self.n_actions + 1), self.success_probabilities))}.')
        print(f'Reward: {reward}.')


class FixedPolicy(Policy):
    def __init__(self, actions):
        self.actions = np.array(actions)

    def __getitem__(self, states):
        return self.actions[len(states) - 1] - 1


class InteractivePolicy(Policy):
    def __getitem__(self, states):
        try:
            return int(input('\nAction: ')) - 1
        except ValueError:
            raise IndexError()


def create_bernoulli_bandits(success_probabilities):
    bb = BernoulliBandits(success_probabilities)

    initial_distribution = np.zeros(bb.n_states)
    initial_distribution[bb.no_reward_state] = 1.

    reward = np.zeros(bb.n_states)
    reward[bb.reward_state] = 1.

    return bb, initial_distribution, reward