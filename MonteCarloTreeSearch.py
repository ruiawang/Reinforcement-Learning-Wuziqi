'''
Implementation of the Monte Carlo Tree Search used in the AlphaGo Zero paper
'''
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
    - Parent (parent node)
    '''
    def __init__(self, parent, P):
        self._parent = parent
        self._P = P
        
        self._children = {} 
        self._N = 0
        self._Q = 0
        self._U = 0
    

    def expand(self, action_priors):
        '''
        expand the tree and create children nodes.
        action_priors: a list of tuples given by (action, prior_probability)
        '''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = Node(self, prob)
    

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
    
class MonteCarloTreeSearch(object):
    '''
    Implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_function, c_puct = 5, n_playout = 400):
        '''
        policy_value_function: function that takes in a board state and outputs (action, probability) tuples 
            as well as a score in [-1,1]
        c_puct: constant that determines level of exploration. higher value means more reliance on priors
        '''
        self._root = Node(None, 1.0)
        self._policy = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout

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
        action_probabilities, leaf_value = self._policy(state)
        has_end, winner = state.end_game()
        if not has_end:
            node.expand(action_probabilities)
        else:
            if winner == -1: # when the game is a tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)

        # updates the values and the visit count of the nodes in the traversed path
        node.recursive_update(-leaf_value)


    def get_move_probabilities(self, state, temperature=1e-3):
        '''
        runs all playouts and returns the available actions and move probabilities.
        state: the current game state
        temperature: (0,1] parameter that controls exploration
        '''
        for i in range(self._n_playout):
            copy_state = copy.deepcopy(state)
            self._playout(copy_state)
        
        action_visit = [(action, node._N) for action, node in self._root._children.items()]
        actions, visit_counts = zip(*action_visit)
        action_probs = softmax(1.0/temperature * np.log(np.array(visit_counts) + 1e-10))

        return actions, action_probs
    
    def update_move(self, last_move):
        '''
        move forward in the tree but maintain other information in subtree
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)
    

    def __str__(self):
        return "Monte Carlo Tree Search"


class MCTSPlayer(object):
    '''
    Implementation of an AI player using Monte Carlo Tree Search
    '''
    def __init__(self, policy_value_function, c_puct = 5, n_playout = 4000, self_play = 0):
        self.mcts = MonteCarloTreeSearch(policy_value_function, c_puct, n_playout)
        self._self_play = self_play
    
    def set_player(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_move(-1)
    

    def get_action(self, board, temperature=1e-3, return_probability=0):
        available_moves = board.available_moves
        
        # pi from AlphaGo Zero paper
        move_probabilities = np.zeros(board.width*board.height)
        if len(available_moves) > 0:
            actions, probabilities = self.mcts.get_move_probabilities(board, temperature)
            move_probabilities[list(actions)] = probabilities
            if self._self_play:
                # use Dirichlet Noise
                move = np.random.choice(actions, p=0.75*probabilities + 0.25*np.random.dirichlet(0.3*np.ones(len(probabilities))))

                # update the tree search with the chosen move
                self.mcts.update_move(move)
            else:
                move = np.random.choice(actions, p=probabilities)
                # reset the MCTS
                self.mcts.update_move(-1)

            if return_probability:
                return move, move_probabilities
            else:
                return move
        else:
            print('Board is full')

    def __str__(self):
        return 'Monte Carlo Tree Search {}'.format(self.player)