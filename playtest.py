'''
Playtest the models as a human
'''

from __future__ import print_function

from game import Board, Game
from MonteCarloTreeSearchBasic import MCTSPlayer as MCTS_Pure
from MonteCarloTreeSearch import MCTSPlayer
from policy_value_network import PolicyValueNetwork


class Human(object):
    """
    blank human player
    """

    def __init__(self):
        self.player = None

    def set_player(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.loc_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.available_moves:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 7, 7
    model_file = 'best_policy.model'
    try:
        board = Board(width=width, height=height, k_in_row=n)
        game = Game(board)

        best_policy = PolicyValueNetwork(width, height, model= model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_function, c_puct=5, n_playout=400)

        human = Human()

        # set start_player=0 for human first
        game.play(human, mcts_player, start_player=0, display=1)
    
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()