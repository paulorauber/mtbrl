from mtbrl.environments.bernoulli_bandits import create_bernoulli_bandits

from mtbrl.environments.bernoulli_bandits import FixedPolicy
from mtbrl.environments.bernoulli_bandits import InteractivePolicy


def main():
    bb, initial_distribution, reward = create_bernoulli_bandits([0.25, 0.75])

    print('\n# Fixed policy simulation')
    policy = FixedPolicy([2] * 10000)
    states = bb.simulate(initial_distribution, reward, policy, render=(len(policy.actions) <= 8))
    print(f'# Fixed policy average reward: {sum(states) / (len(states) - 1)}')

    print('\n# Interactive policy simulation')
    policy = InteractivePolicy()
    states = bb.simulate(initial_distribution, reward, policy)
    if len(states) > 1:
        print(f'# Interactive policy average reward: {sum(states) / (len(states) - 1)}')


if __name__ == '__main__':
    main()