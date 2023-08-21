'''
Implementation of Policy Value Network in AlphaGo Zero style in TensorFlow
'''
from __future__ import print_function

import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras.layers import Conv2D, Activation, Dense, Flatten, Add, BatchNormalization
from keras.regularizers import L2
from keras.optimizers import Adam

class PolicyValueNetwork():
    '''
    Implementation of Policy Value Network in AlphaGo Zero style
    Has simplified architecture in the interest of time and speed, but still maintains the structure
    '''
    def __init__(self, width, height, model=None):
        self.width = width
        self.height = height
        self.L2 = 1e-4 # L2 coefficient used in AlphaGo Zero paper
        
        self.create_network()
        self._loss_train_operation()

        if model:
            params = pickle.load(open(file=model, mode='rb'))
            self.model.set_weights(params)
    
    def create_network(self):
        input_x = network = Input((4, self.width, self.height))

        # convolution layers
        network = Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format='channels_last', activation='relu',kernel_regularizer=L2(self.L2))(network)
        network = Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format='channels_last', activation='relu',kernel_regularizer=L2(self.L2))(network)
        network = Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_last', activation='relu',kernel_regularizer=L2(self.L2))(network)

        # policy head
        policy_network = Conv2D(filters=4, kernel_size=(1,1), data_format='channels_last', activation='relu', kernel_regularizer=L2(self.L2))(network)
        policy_network = Flatten()(policy_network)
        self.policy_network = Dense(self.width*self.height, activation='softmax', kernel_regularizer=L2(self.L2))(policy_network)

        # value head
        value_network = Conv2D(filters=2, kernel_size=(1,1), data_format='channels_last', activation='relu', kernel_regularizer=L2(self.L2))(network)
        value_network = Flatten()(value_network)
        value_network = Dense(64, kernel_regularizer=L2(self.L2))(value_network)
        self.value_network = Dense(1, activation='tanh', kernel_regularizer=L2(self.L2))(value_network)

        self.model = Model(input_x, [self.policy_network, self.value_network])


        def policy_value(state):
            state_input_batch = np.array(state)
            results = self.model.predict_on_batch(state_input_batch)
            return results
        
        self.policy_value = policy_value
    
    def policy_value_function(self, board):
        '''
        takes in a board and outputs list of tuples (action, probability) for all available actions and the score of the board state
        '''
        available_moves = board.available_moves
        current_board = board.current_board()
        action_probabilities, value = self.policy_value(current_board.reshape(-1, 4, self.width,self.height))
        action_probabilities = zip(available_moves, action_probabilities.flatten()[available_moves])

        return action_probabilities, value[0][0]

    def _loss_train_operation(self):
        '''
        Loss is calculated via: L = (z-v)^2 + pi^T * ln(p) + c||theta||^2
        '''
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mean_squared_error'])

        def self_entropy(probabilities):
            return -np.mean(np.sum(probabilities*np.log(probabilities + 1e-10), axis=1))
        
        def train_step(state_in, mcts_probabilities, winner, lr):
            state_input_batch = np.array(state_in)
            mcts_probs_batch = np.array(mcts_probabilities)
            winner_batch = np.array(winner)

            loss = self.model.evaluate(state_input_batch, [mcts_probs_batch, winner_batch], batch_size=len(state_in), verbose=0)
            action_probabilities, _ = self.model.predict_on_batch(state_input_batch)

            entropy = self_entropy(action_probabilities)

            keras.backend.set_value(self.model.optimizer.lr, lr)

            self.model.fit(state_input_batch, [mcts_probs_batch, winner_batch], batch_size=len(state_in), verbose=0)

            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_params(self):
        policy_params = self.model.get_weights()
        return policy_params
    
    def save_model(self, model):
        network_params = self.get_policy_params()
        pickle.dump(network_params, open(model, 'wb'), protocol=2)