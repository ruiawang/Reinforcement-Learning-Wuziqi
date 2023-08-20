'''
Implementation of Policy Value Network in AlphaGo Zero style
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class Net(nn.Module):
    '''
    Simplified implementation of AlphaGo Zero's Policy Value Neural Network
    Still maintains the overall structure, but the architecture is simplified in the interest of speed/time.
    '''
    def __init__(self, width, height):
        super(Net, self).__init__()

        # board dimensions
        self.width = width
        self.height = height

        # convolution layers
        self.conv_1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # policy layers
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4*width*height,width*height)

        # value layers
        self.value_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.value_fc_1 = nn.Linear(2*width*height, 32)
        self.value_fc_2 = nn.Linear(32, 1)
    
    def forward(self, state):
        # convolution layers
        v = F.relu(self.conv_1(state))
        v = F.relu(self.conv_2(v))
        v = F.relu(self.conv_3(v))

        # policy layers
        v_policy = F.relu(self.policy_conv(v))
        v_policy = v_policy.view(-1, 4*self.width*self.height)
        v_policy = F.log_softmax(self.policy_fc(v_policy))

        # value layers
        v_value = F.relu(self.value_conv(v))
        v_value = v_value.view(-1, 2*self.width*self.height)
        v_value = F.relu(self.value_fc_1(v_value))
        v_value = F.tanh(self.value_fc_2(v_value))

        return v_policy, v_value

def set_lr(optimizer, lr):
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PolicyValueNetwork():
    '''
    Policy Value Network Implementation
    '''
    def __init__(self, width, height, model=None):
        self.width = width
        self.height = height
        self.l2 = 1e-4 # the l2 value in AlphaGo Zero paper
        
        self.policy_value_net = Net(width, height)
    
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2)

        # if pre-load model
        if model:
            params = torch.load(model)
            self.policy_value_net.load_state_dict(params)
    
    def policy_value(self, state_batch):
        '''
        Inputs board states and returns their action probabilities and state values
        '''
        state_batch = Variable(torch.FloatTensor(state_batch))
        
        # run the policy value network to evalute their action probabilities and state values
        action_probabilities_log, value = self.policy_value_net(state_batch)
        action_probabilities = np.exp(action_probabilities_log.data.numpy()) # normalize the log values
        
        return action_probabilities, value.data.numpy()
    
    def policy_value_function(self, board):
        '''
        Outputs action probabilities and state values for all possible actions on the board
        '''
        available_moves = board.available_moves

        current_state = np.ascontiguousarray(board.current_board().reshape(-1, 4, self.width,self.height))

        action_probabilities_log, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
        action_probabilities = np.exp(action_probabilities_log.data.numpy().flatten())

        action_probabilities = zip(available_moves, action_probabilities[available_moves])

        value = value.data[0][0]

        return action_probabilities, value

    def train_step(self, state_batch, mcts_probabilities, winner_batch, lr):
        '''performs a training step'''
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probabilities = Variable(torch.FloatTensor(mcts_probabilities))
        winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad() # reset the optimizer gradient to 0
        set_lr(self.optimizer,lr) # set the learning rate

        action_probabilities_log, value = self.policy_value_net(state_batch)

        # implementation of losses used in AlphaGo Zero paper (mse and cross-entropy loss)
        value_loss = F.mse_loss(value.view(-1), winner_batch)

        # policy loss is defined as (z-v)^2 - pi^T * log(p) + c||theta||^2.
        # the l2 penalty term is already incorporated within the optimizer 
        policy_loss = -torch.mean(torch.sum(mcts_probabilities*action_probabilities_log, 1))

        total_loss = value_loss + policy_loss
    
        # step backwards and optimize loss
        total_loss.backward()
        self.optimizer.step()

        # entropy
        policy_entropy = -torch.mean(torch.sum(torch.exp(action_probabilities_log)*action_probabilities_log,1))

        return total_loss.item(), policy_entropy.item()
    
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model)