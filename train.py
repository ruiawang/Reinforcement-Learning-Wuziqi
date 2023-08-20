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

    def extend_data(self, play_data):
        '''
        for play data which is list of tuples (state, monte carlo tree search prob, winner), extend the dataset by rotating and flipping it
        '''
        extended_data = []
        for state, mcts_probabilities, winner in play_data:
            for i in range(1,5): # have a rotated state and flipped rotated state that can be 4 times
                rotated_state = np.array([np.rot90(s, i) for s in state])
                rotated_mcts_probs = np.rot90(np.flipud(mcts_probabilities.reshape(self.width, self.height)), i)
                
                extended_data.append((rotated_state, np.flipud(rotated_mcts_probs).flatten()), winner)

                # 
                flipped_state = np.array([np.fliplr(s) for s in rotated_state])
                flipped_mcts_probs = np.fliplr(rotated_mcts_probs)

                extended_data.append((flipped_state, np.flipud(flipped_mcts_probs).flatten(), winner))
        
        return extended_data

    def collect_self_play(self, num_games=1):
        '''
        collect self-play data
        '''
        for i in range(num_games):
            winner, play_data = self.Game.self_play(self.MCTS_Player, temperature=self.temperature)
            play_data = list(play_data)[:]
            self.play_length = len(play_data)

            # add self-play and extended play data to buffer
            extended_play_data = self.extend_data(play_data)
            self.buffer.extend(extended_play_data)
    
    def update_policy_value_network(self):
        '''
        update policy value network
        '''