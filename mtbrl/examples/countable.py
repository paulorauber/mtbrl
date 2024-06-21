import numpy as np

from mtbrl.environments.bernoulli_bandits import create_bernoulli_bandits
from mtbrl.environments.bernoulli_bandits import FixedPolicy as FixedPolicyBB

from mtbrl.environments.gridworld import create_gridworld
from mtbrl.environments.gridworld import FixedPolicy as FixedPolicyGW

from mtbrl.algorithms.bamdp import CountableBAMDP


def bandits_test():
    print('\n# Bandits test')

    bb1, initial_distribution, reward = create_bernoulli_bandits([1., 0., 0., 0.])
    bb2, initial_distribution, reward = create_bernoulli_bandits([0., 1., 0., 0.])
    bb3, initial_distribution, reward = create_bernoulli_bandits([0., 0., 1., 0.])
    bb4, initial_distribution, reward = create_bernoulli_bandits([0., 0., 0., 1.])

    bamdp = CountableBAMDP([bb1, bb2, bb3, bb4], np.ones(4)/4, reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 6

    policy = FixedPolicyBB([4] * horizon)
    print('## Fixed policy simulation:')
    bb4.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    bb4.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def corridors_test():
    print('\n# Corridors test')
    gw1, initial_distribution, reward = create_gridworld([['.', '&', '.', '.', '.', '$']], 0.)
    gw2, initial_distribution, reward = create_gridworld([['$', '&', '.', '.', '.', '.']], 0.)

    bamdp = CountableBAMDP([gw1, gw2], [0.5, 0.5], reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 7

    policy = FixedPolicyGW('adddddd')
    print('## Fixed policy simulation:')
    gw1.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    gw1.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def gridworlds_test():
    print('\n# Grid worlds test')
    gw1, initial_distribution, reward = create_gridworld([['&', '.', '.'],
                                                          ['.', '#', '.'],
                                                          ['.', '.', '$']], 0.)

    gw2, initial_distribution, reward = create_gridworld([['&', '.', '.'],
                                                          ['.', '#', '.'],
                                                          ['$', '.', '.']], 0.)

    gw3, initial_distribution, reward = create_gridworld([['&', '.', '$'],
                                                          ['.', '#', '.'],
                                                          ['.', '.', '.']], 0.)

    bamdp = CountableBAMDP([gw1, gw2, gw3], np.ones(3)/3, reward, discount=0.99)
    print(f'## Effective horizon: {bamdp.effective_horizon(epsilon=1e-8)}')

    horizon = 7

    policy = FixedPolicyGW('ddssaaw')
    print('## Fixed policy simulation:')
    gw2.simulate(initial_distribution, reward, policy)
    print(f'## Fixed policy value: {bamdp.value(initial_distribution, policy, horizon=horizon)}')

    optimal_policy = bamdp.solve(initial_distribution, horizon)
    print('\n## Optimal policy simulation:')
    gw3.simulate(initial_distribution, reward, optimal_policy)
    print(f'## Optimal policy value: {bamdp.value(initial_distribution, optimal_policy, horizon=horizon)}')


def main():
    bandits_test()
    corridors_test()
    gridworlds_test()


if __name__ == '__main__':
    main()