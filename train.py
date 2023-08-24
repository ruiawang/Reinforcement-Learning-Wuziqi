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
        # change to alter game
        self.width = 6 # m
        self.height = 6 # n
        self.k_in_row = 4 # k
        self.Board = Board(width=self.width, height=self.height, k_in_row=self.k_in_row)
        self.Game = Game(self.Board)

        # training params
        self.lr = 2e-3
        self.lr_mult = 1.0 # changing learning rate based on kl, initally at 1
        self.temperature = 1.0
        self.n_playout = 400
        self.c_puct = 5

        self.buffer_size = 10000
        self.batch_size = 512
        self.buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        
        self.epochs = 5 # how many training steps for each update
        self.kl_target = 0.02
        self.check_frequency = 25
        self.game_batch_number = 750
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
        We are using the fact that Wuziqi (in square boards) is invariant to rotation and flipping.
        This fact was used in the AlphaGo Zero paper since Go is also invariant to rotation/flips.

        This allows us to generate much more self-play data and help the diversity of the data.
        Generating self-play data is the bottleneck in the computing process, so doing this allows us to speed up the process more.
        for play data which is list of tuples (state, monte carlo tree search prob, winner), extend the dataset by rotating and flipping it
        '''
        extended_data = []
        for state, mcts_probabilities, winner in play_data:
            for i in [1,2,3,4]: # have a rotated state and flipped rotated state that can be 4 times
                rotated_state = np.array([np.rot90(s, i) for s in state])
                rotated_mcts_probs = np.rot90(np.flipud(mcts_probabilities.reshape(self.width, self.height)), i)
                
                extended_data.append((rotated_state, np.flipud(rotated_mcts_probs).flatten(), winner))

                # flip the state across mirror
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
        small_batch = random.sample(self.buffer, self.batch_size)
        # small_batch is a (random) sample of a list of tuples (state, mcts_prob, winner)
        state_batch = [data[0] for data in small_batch]
        mcts_probability_batch = [data[1] for data in small_batch]
        winner_batch = [data[2] for data in small_batch]

        prev_act_probs, prev_value = self.policy_value_network.policy_value(state_batch)

        for i in range(self.epochs):
            # train the policy value neural network
            loss, entropy = self.policy_value_network.train_step(state_batch, mcts_probability_batch, winner_batch, self.lr*self.lr_mult)
            # new action probs and value after training the policy value nn
            new_act_probs, new_value = self.policy_value_network.policy_value(state_batch)

            # kl-divergence between the previous and new probs
            kl = np.mean(np.sum(prev_act_probs*(np.log(prev_act_probs + 1e-10) - np.log(new_act_probs + 1e-10)), axis=1))

            # stop early if we have diverged a lot
            if kl > self.kl_target*4:
                break
        
        # change learning rate based on kl
        if kl > self.kl_target*2 and self.lr_mult > 0.1:
            self.lr_mult /= 1.5
        elif kl < self.kl_target/2 and self.lr_mult < 10:
            self.lr_mult *= 1.5
        
        # explained variation
        explained_var_prev = (1 - np.var(np.array(winner_batch) - prev_value.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_value.flatten()) / np.var(np.array(winner_batch)))

        print('kl:{:.5f}, loss:{}, entropy:{}, lr_mult:{:.3f}, explained_var_prev:{:.3f}, explained_var_new:{:.3f}'.format(
            kl, loss, entropy, self.lr_mult, explained_var_prev, explained_var_new))

        return loss, entropy
    
    def evaluate_policy(self, num_games = 10):
        '''
        evaluate policy by playing against the basic MCTS player and return win rate
        '''
        mcts_player_current = MCTSPlayer(self.policy_value_network.policy_value_function, c_puct=self.c_puct, n_playout=self.n_playout)
        mcts_player_basic = Basic_MCTS_Player(c_puct=5, n_playout=self.basic_mcts_n_playout)

        win_dict = defaultdict(int)
        for i in range(num_games):
            winner = self.Game.play(mcts_player_current, mcts_player_basic, start_player= i % 2, display=0) # alternate who starts
            win_dict[winner] += 1
        winrate = 1.0*(win_dict[1] + 0.5*win_dict[-1]) / num_games

        print('num_playouts:{}, win: {}, lose: {}, tie:{}'.format(self.basic_mcts_n_playout, win_dict[1], win_dict[2], win_dict[-1]))
        return winrate
    
    def run(self):
        '''
        Runs the Training Pipeline
        '''
        try:
            for i in range(self.game_batch_number):
                self.collect_self_play(self.play_batch_size)
                print('batch i:{}, play_length:{}'.format(i+1, self.play_length))
                
                if len(self.buffer) > self.batch_size:
                    loss, entropy = self.update_policy_value_network()
                
                # evaluates the model every check_frequency  batches
                if (i+1) % self.check_frequency == 0:
                    print('current batch:{}'.format(i+1))

                    winrate = self.evaluate_policy()

                    self.policy_value_network.save_model('./current_policy.model')

                    if winrate > self.best_win_rate:
                        print("NEW BEST POLICY")
                        self.best_win_rate = winrate
                        # update the best_policy
                        self.policy_value_network.save_model('./best_policy.model')
                        if (self.best_win_rate == 1.0 and self.basic_mcts_n_playout < 5000):
                            # increase the number of playouts basic MCTS gets if only winning
                            self.basic_mcts_n_playout += 1000
                            self.best_win_rate = 0.0
        except KeyboardInterrupt:
            print('\n\rquit out')

if __name__ == '__main__':
    training_pipeline = Training_Pipeline()
    training_pipeline.run()