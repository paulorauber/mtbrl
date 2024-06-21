import numpy as np


class Policy:
    def __getitem__(self, states):
        raise NotImplementedError()


class Model:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

    def __call__(self, s, a, next_s):
        raise NotImplementedError()

    def render(self, state, reward):
        raise NotImplementedError()

    def simulate(self, initial_distribution, reward, policy, render=True, seed=None):
        random_state = np.random.RandomState(seed)
        states = [random_state.choice(self.n_states, p=initial_distribution)]

        if render:
            self.render(states[-1], reward[states[-1]])

        while True:
            try:
                a = policy[tuple(states)]
            except LookupError:
                break

            probabilities = [self(states[-1], a, next_s) for next_s in range(self.n_states)]
            states.append(random_state.choice(self.n_states, p=probabilities))

            if render:
                self.render(states[-1], reward[states[-1]])

        return states


class OptimalPolicy(Policy):
    def __init__(self, q_star):
        self.q_star = q_star
        self._policy = {}

    def __getitem__(self, states):
        if states not in self._policy:
            actions = tuple([self[states[:t]] for t in range(1, len(states))])
            self._policy[states] = np.argmax(self.q_star[(states, actions)])

        return self._policy[states]


class BAMDP:
    def __init__(self, n_states, n_actions, reward, discount):
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward = reward
        self.discount = discount

    def posterior_predictive(self, initial_distribution, states, actions):
        raise NotImplementedError()

    def effective_horizon(self, epsilon):
        c = np.max(np.abs(self.reward))
        return int(np.log(( epsilon * (1 - self.discount) ) / (2 * c)) / np.log(self.discount) + 1)

    def value(self, initial_distribution, policy, horizon, states=[], actions=[]):
        v = 0.
        if len(states) == 0:
            for state, probability in enumerate(initial_distribution):
                if not np.allclose(probability, 0):
                    v += probability * self.value(initial_distribution, policy, horizon, [state], [])
        elif len(states) - 1 < horizon:
            actions = actions + [policy[tuple(states)]]
            probabilities = self.posterior_predictive(initial_distribution, states, actions)
            for state, probability in enumerate(probabilities):
                if not np.allclose(probability, 0):
                    next_v = self.value(initial_distribution, policy, horizon, states + [state], actions)
                    v += probability * (self.reward[state] + self.discount * next_v)

        return v

    def optimal_values(self, initial_distribution, horizon, states=[], actions=[], q_star=None):
        if len(states) == 0:
            q_star = {}
            for state, probability in enumerate(initial_distribution):
                if not np.allclose(probability, 0):
                    self.optimal_values(initial_distribution, horizon, [state], [], q_star)

            return q_star

        action_values = np.zeros(self.n_actions)

        if len(states) - 1 < horizon:
            for action in range(self.n_actions):
                probabilities = self.posterior_predictive(initial_distribution, states, actions + [action])
                for state, probability in enumerate(probabilities):
                    if not np.allclose(probability, 0):
                        next_values = self.optimal_values(initial_distribution, horizon, states + [state], actions + [action], q_star)
                        action_values[action] += probability * (self.reward[state] + self.discount * np.max(next_values))

            q_star[(tuple(states), tuple(actions))] = action_values

        return action_values

    def solve(self, initial_distribution, horizon):
        return OptimalPolicy(self.optimal_values(initial_distribution, horizon))


class CountableBAMDP(BAMDP):
    def __init__(self, models, prior, reward, discount):
        BAMDP.__init__(self, models[0].n_states, models[0].n_actions, reward, discount)
        self.models = models
        self.prior = prior

    def posterior_predictive(self, initial_distribution, states, actions):
        probability_trajectory = np.zeros(len(self.models))
        for i, p in enumerate(self.models):
            probability_trajectory[i] = initial_distribution[states[0]] * self.prior[i]

            for j in range(len(states) - 1):
                probability_trajectory[i] *= p(states[j], actions[j], states[j + 1])

        posterior_probability = np.zeros(self.n_states)

        c = np.sum(probability_trajectory)
        if np.allclose(c, 0):
            posterior_probability[0] = 1.
        else:
            for state in range(self.n_states):
                for i, p in enumerate(self.models):
                    posterior_probability[state] += probability_trajectory[i] * p(states[-1], actions[-1], state)

            posterior_probability = posterior_probability / c

        return posterior_probability


class DirichletBAMDP(BAMDP):
    def __init__(self, alphas, reward, discount):
        BAMDP.__init__(self, alphas.shape[0], alphas.shape[1], reward, discount)
        self.alphas = alphas

    def posterior_predictive(self, initial_distribution, states, actions):
        posterior_probability = np.zeros(self.n_states)

        if np.allclose(initial_distribution[states[0]], 0):
            posterior_probability[0] = 1.
        else:
            last_s, last_a = states[-1], actions[-1]
            posterior_probability += self.alphas[last_s, last_a]

            for j in range(len(states) - 1):
                if (states[j], actions[j]) == (last_s, last_a):
                    posterior_probability[states[j + 1]] += 1.

            posterior_probability = posterior_probability / np.sum(posterior_probability)

        return posterior_probability
