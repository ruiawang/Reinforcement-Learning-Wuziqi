import numpy as np

class Board(object):
    '''
    Implementation of the board for the game.
    Wuziqi is an example of a m,n,k-game, played on a m x n board, and the goal is to connect k in a row.
    Specifically, Wuziqi is traditionally 15,15,5. (tic-tac-toe is 3,3,3).
    Our default will be a 7,7,5 game out of interest for computation and training time.
    '''
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 7)) # m
        self.height = int(kwargs.get('height', 7)) # n
        self.k_in_row = int(kwargs.get('k_in_row', 5)) # k

        self.players = [1, 2]

        # store board states as a dictionary with key being a move indicating position, and value being the player
        self.states = {}
        
    def init_board(self, start_player=0):
        if self.width < self.k_in_row or self.height < self.k_in_row:
            raise Exception('board width and height cannot be less than {}'.format(self.k_in_row))
        
        self.current_player = self.players[start_player] # player 1 starts
        
        # at the start there are m * n possible moves
        self.available_moves = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
    
    '''
    We can store a move/location as a single integer rather than a coordinate pair due to the rectangular nature of the board.
    For example in a 2x4 board:
    4 5 6 7
    0 1 2 3
    or a 2x3 board:
    3 4 5
    0 1 2
    
    the move 5 represents position (1,1) in a 2x4 board and (1,2) in a 2x3 board
    this is more intuitive than the traditional cartesian reading of (horizontal, vertical)
    as you find the right row and then go through it to find the right position
    Any move z is represented by a coordinate by: (z//m, z%m)
    Inversely, any coordinate location (h,w) represents the move h*m + w
    '''
    def move_to_loc(self, move):
        h = move // self.width
        w = move % self.width
        return [h,w]
    

    def loc_to_move(self, location):
        if len(location) != 2:
            return -1
        
        h = location[0]
        w = location[1]
        move = h * self.width + w
        
        if move not in range(self.width*self.height):
            return -1
        
        return move

    
    def current_board(self):
        '''
        return current board state, from the perspective of the current player
        '''
