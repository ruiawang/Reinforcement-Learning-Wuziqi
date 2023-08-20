'''
Implementation of AlphaGo Zero training pipeline
'''

from __future__ import print_function
import numpy as np
import random
from collections import defaultdict, deque

from game import Board, Game
from policy_value_network import PolicyValueNetwork
from MonteCarloTreeSearch import MCTSPlayer
from MonteCarloTreeSearchBasic import MCTSPlayer as Basic_MCTS_Player


class Training_Pipeline():
    def __init__(self, saved_model=None):
        # Board/Game params
        self.width = 6 # m
        self.height = 6 # n
        self.k_in_row = 4
        self.Board = Board(width=self.width, height=self.height, k_in_row=self.k_in_row)
        self.Game = Game(self.Board)

        # training params
        self.lr = 2e-3
        self.lr_mult = 1.0 # changing learning rate based on kl
        self.temperature = 1.0
        self.n_playout = 400
        self.c_puct = 2

        self.buffer_size = 10000
        self.batch_size = 512
        self.buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        
        self.epochs = 5 # how many training steps for each update
        self.kl_target = 0.02
        self.check_frequency = 50
        self.game_batch_number = 1500
        self.best_win_rate = 0.0

        # how many playouts for the basic mcts player (opponent)
        self.basic_mcts_n_playout = 1000

        if saved_model: # use saved network/model to start training
            self.policy_value_network = PolicyValueNetwork(self.width, self.height, model=saved_model)
        else: # create new policy value network
            self.policy_value_network = PolicyValueNetwork(self.width, self.height, None)
        
        self.MCTS_Player = MCTSPlayer(self.policy_value_network.policy_value_function, self.c_puct, self.n_playout, self_play=1)
        
