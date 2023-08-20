'''
Basic implementation of Monte Carlo Tree Search (serves as opponent against the trained MCTS with policy-value network)
'''
import copy
from operator import itemgetter
import numpy as np

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
            self._parent.recursive_update(-leaf_value)
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

def simulation_function(board):
    # randomly simulate/rollout a move
    action_probabilities = np.random.rand(len(board.available_moves))
    return zip(board.available_moves, action_probabilities)

def policy_value_function(board):
    '''
    pure MCTS policy value function is different than Alpha(Go) Zero style MCTS
    Alpha(Go) Zero MCTS used policy function with action probabilities and a score from [-1,1]
    pure MCTS has action probability be uniform, with score 0.
    '''
    action_probabilities = np.ones(len(board.available_moves))/len(board.available_moves)
    return zip(board.available_moves, action_probabilities), 0

class MonteCarloTreeSearch(object):
    '''
    Implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_function, c_puct = 5, n_playout = 1000):
        '''
        policy_value_function: function that takes in a board state and outputs (action, probability) tuples 
            as well as a score in [-1,1]
        c_puct: constant that determines level of exploration. higher value means more reliance on priors
        '''
        self._root = Node(None, 1.0, -1) # Since the starting player is 1 (black)
        self._policy = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _evaluate_simulation(self, state, sim_limit=500):
        player = state.get_current_player()

        for i in range(sim_limit):
            has_end, winner = state.end_game()
            if has_end:
                break
            
            action_probabilities = simulation_function(state)
            max_probability_action = max(action_probabilities, key=itemgetter(1))[0]
            state.do_move(max_probability_action)
        else:
            # hit limit
            print('simulation reached move limit')
        
        if winner == 0:
            return 0
        return 1 if winner == player else -1

    def _playout(self, state):
        '''
        runs a single playout from root to leaf, and getting the value at the leaf, backpropagating it through parents
        '''
        node = self._root
        while True:
            if node.is_leaf():
                break
        
            action, node = node.select(self._c_puct)
            state.do_move(action)
    
        '''
        evaluates the leaf using network that also outputs tuple of action probabilities as well as score {-1,1} for current player
        checks for the end of the game and then returns the leaf value recursively 
        '''
        action_probabilities, _ = self._policy(state)
        has_end, winner = state.end_game()
        if not has_end:
            node.expand(action_probabilities, state.current_player)
        
        '''
        Key difference between AlphaZero/AlphaGo Zero MCTS and pure MCTS is the simulation step. AlphaGo Zero MCTS simulation step is by
        evaluation of policy value neural network.
        '''
        leaf_value = self._evaluate_simulation(state)

        # updates the values and the visit count of the nodes in the traversed path
        node.recursive_update(-leaf_value)

    def get_move(self, state):
        '''
        run all playouts and return most visited action
        '''
        for i in range(self._n_playout):
            copy_state = copy.deepcopy(state)
            self._playout(copy_state)
        
        return max(self._root._children.items(), key=lambda action_node: action_node[1]._N)[0]

    def update_move(self, last_move):
        '''
        move forward in the tree but maintain other information in subtree
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0, -1)

    def __str__(self):
        return "Monte Carlo Tree Search"

class MCTSPlayer(object):
    def __init__(self, c_puct=2, n_playout=1000):
        self.mcts = MonteCarloTreeSearch(policy_value_function, c_puct, n_playout)

    def set_player(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_move(-1)

    def get_action(self, board):
        available_moves = board.available_moves
        if len(available_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_move(-1)
            return move
        else:
            print('Board is full')

    def __str__(self):
        return "Monte Carlo Tree Search {}".format(self.player)