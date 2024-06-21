from mtbrl.environments.gridworld import create_gridworld

from mtbrl.environments.gridworld import FixedPolicy
from mtbrl.environments.gridworld import InteractivePolicy


def main():
    worlds = {'large': [['&', '.', '.', '.', '.', '.', '.', '.'],
                        ['.', '.', '.', '.', '.', '.', '.', '.'],
                        ['.', '.', '.', '#', '.', '.', '.', '.'],
                        ['.', '.', '.', '.', '.', '#', '.', '.'],
                        ['.', '.', '.', '#', '.', '.', '.', '.'],
                        ['.', '#', '#', '.', '.', '.', '#', '.'],
                        ['.', '#', '.', '.', '#', '.', '#', '.'],
                        ['.', '.', '.', '#', '.', '.', '.', '$']],

              'medium': [['&', '.', '.', '.'],
                         ['.', '#', '.', '#'],
                         ['.', '.', '.', '#'],
                         ['#', '.', '.', '$']],

              'small': [['&', '.', '.'],
                        ['.', '#', '.'],
                        ['.', '.', '$']]}

    gw, initial_distribution, reward = create_gridworld(worlds['medium'], 0.)

    print('\n# Fixed policy simulation')
    policy = FixedPolicy('ddsssdw')
    gw.simulate(initial_distribution, reward, policy)

    print('\n# Interactive policy simulation')
    policy = InteractivePolicy()
    gw.simulate(initial_distribution, reward, policy)


if __name__ == '__main__':
    main()
