from argparse import ArgumentParser


class BaseOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--algorithm', type=str, default='ValueItration')
        self.parser.add_argument('--n_rounds', type=int, default=500, help='Number of rounds')
        self.parser.add_argument('--ub_gamma', type=float, default=1, help='upper bound of discount factor')
        self.parser.add_argument('--lb_gamma', type=float, default=0, help='lower bound of discount factor')
        self.parser.add_argument('--NA', type=int, default=6, help='Length of Actions Space')
        self.parser.add_argument('--NS', type=int, default=500, help='Length of States Space')
        self.parser.add_argument('--end_delta', type=float, default=0.00001, help='end delta')
        self.parser.add_argument('--print_interval', type=int, default=50, help='print interval')

    def parse(self):
        return self.parser.parse_args()