import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class Net(nn.Module):
    '''
    Simplified implementation of AlphaGo Zero's Policy Value Network
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
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
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
