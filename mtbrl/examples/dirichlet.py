import numpy as np

from mtbrl.environments.bernoulli_bandits import create_bernoulli_bandits
from mtbrl.environments.bernoulli_bandits import FixedPolicy as FixedPolicyBB

from mtbrl.environments.gridworld import create_gridworld
from mtbrl.environments.gridworld import FixedPolicy as FixedPolicyGW

from mtbrl.algorithms.bamdp import DirichletBAMDP


def alphas_from_models(models):
    alphas = np.full((models[0].n_states, models[0].n_actions, models[0].n_states), 1e-8)

    for state in range(models[0].n_states):
        for action in range(models[0].n_actions):
            all_equal = True
            for i in range(1, len(models)):
                if not np.allclose(models[0]._p[state, action], models[i]._p[state, action]):
                    all_equal = False

            if all_equal:
                alphas[state, action] += models[0]._p[state, action] * 1e8
            else:
                for i in range(len(models)):
                    alphas[state, action] += models[i]._p[state, action] / len(models)

    return alphas


def bandits_test():
    print('\n# Bandits test')

    bb1, initial_distribution, reward = create_bernoulli_bandits([1., 0.])
    bb2, initial_distribution, reward = create_bernoulli_bandits([0., 1.])

    alphas = alphas_from_models([bb1, bb2])
    bamdp = DirichletBAMDP(alphas, reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 8

    policy = FixedPolicyBB([1] * horizon)
    print('## Fixed policy simulation:')
    bb1.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    bb2.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def corridors_test():
    print('\n# Corridors test')
    gw1, initial_distribution, reward = create_gridworld([['.', '&', '$']], 0.)
    gw2, initial_distribution, reward = create_gridworld([['$', '&', '.']], 0.)

    alphas = alphas_from_models([gw1, gw2])
    bamdp = DirichletBAMDP(alphas, reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 7

    policy = FixedPolicyGW('adddddd')
    print('## Fixed policy simulation:')
    gw1.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    gw2.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def gridworlds_test():
    print('\n# Grid worlds test')
    gw1, initial_distribution, reward = create_gridworld([['&', '.', '.'],
                                                          ['.', '#', '.'],
                                                          ['.', '.', '$']], 0.)

    gw2, initial_distribution, reward = create_gridworld([['&', '.', '.'],
                                                          ['.', '#', '.'],
                                                          ['$', '.', '.']], 0.)

    alphas = alphas_from_models([gw1, gw2])
    bamdp = DirichletBAMDP(alphas, reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 8

    policy = FixedPolicyGW('ssddwwww')
    print('## Fixed policy simulation:')
    gw1.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    gw1.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def main():
    bandits_test()
    corridors_test()
    gridworlds_test()


if __name__ == '__main__':
    main()

