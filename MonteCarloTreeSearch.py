import numpy as np
import copy


def softmax(x):
    '''
    keras implementation of softmax
    subtracting by the max logit allows for numerical stability,
    without altering the function
    '''
    p = np.exp(x - np.max(x))
    p /= np.sum(p)
    return p


class Node(object):
    '''
    A node in the Monte Carlo Tree Search tree.
    
    Each node represents a board state.

    Each node contains the information:
    - P: the prior probability of being selected by its parent
    - N: how many times its visited during the search
    - Q: the total value from this node across all visits
    - U: visit-count adjusted prior score/Upper confidence Bound 
    - Children (map where an action is mapped to another Node)
    - Whose turn it is 
    '''
    def __init__(self, parent, P, current_player):
        self._parent = parent
        self._P = P
        
        self._children = {} 
        self._N = 0
        self._Q = 0
        self._U = 0
        self._player= current_player
        self._tree_player = 1
    

    def expand(self, action_priors, current_player):
        '''
        expand the tree and create children nodes.
        action_priors: a list of tuples given by (action, prior_probability)
        '''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = Node(self, prob, current_player)
    

    def select(self, c_puct):
        """
        select the action/node that results in the maximum Q + U value.
        returns a tuple of (action, next node)
        """
        return max(self._children.items(), key=lambda action_node: action_node[1].get_value(c_puct))
    
    def get_value(self, c_puct):
        '''
        calculates value of the node, with c_puct parameter.

        the total value is given by the value Q + U
        where U is calculated by U = c_puct * P * sqrt(N_parent) / (1 + N)
        '''
        self._U = c_puct * self._P * np.sqrt(self._parent._N) / (1 + self._N)

        return self._Q + self._U

    def update(self, leaf_value):
        '''
        update node values based on evaluation of leafs.
        leaf_value: value of the evaluation of the subtree
        '''
        self._N += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._N
    
    def recursive_update(self, leaf_value):
        '''
        recursively update nodes, ancestors first
        '''
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        '''
        check if a node is a leaf (i.e it has no children expanded)
        '''
        return self._children == {}


    def is_root(self):
        '''
        check if a node is a root (i.e it has no parent)
        '''
        return self._parent is None
    
class MonteCarloTreeSearch(object):
    '''
    Implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_function, c_puct = 2, n_playout = 1000):
        '''
        policy_value_function: function that takes in a board state and outputs (action, probability) tuples 
            as well as a score in [-1,1]
        c_puct: constant that determines level of exploration. higher value means more reliance on priors
        '''
        self._root = Node(None, 1.0, 2) # Since the starting player is 1
        self._policy = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        '''
        runs a single playout from root to leaf, and getting the value at the leaf, backpropagating it through parents
        '''
    
    def get_move_probabilities(self, state, temperature):
        '''
        runs all playouts and returns the available actions and move probabilities.
        state: the current game state
        temperature: (0,1] parameter that controls exploration
        '''