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

        self.players = [1, 2] # 1 is black, 2 is white

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
    For example in a 4x2 board:
    4 5 6 7
    0 1 2 3
    or a 3x2 board:
    3 4 5
    0 1 2
    
    the move 5 represents position (1,1) in a 4x2 board and (1,2) in a 3x2 board
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

        # 4 m*n grids to represent the state.
        square_state = np.zeros((4,self.width,self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_current = moves[players == self.current_player]
            move_opponent = moves[players != self.current_player]
            
        
            square_state[0][move_current // self.width, move_current % self.height] = 1.0
            square_state[1][move_opponent // self.width, move_opponent % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:,:] = 1.0
        
        return square_state[:,::-1, :]
    
    def do_move(self, move):
        self.states[move] = self.current_player
        self.available_moves.remove(move)
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = move
    
    
    def has_winner(self):
        width = self.width
        height = self.height
        states = self.states
        k = self.k_in_row

        moved_pos = list(set(range(width*height)) - set(self.available_moves))

        if len(moved_pos) < self.k_in_row*2-1:
            return False, -1
        
        for m in moved_pos:
            h = m // width
            w = m % width
            player = states[m]
            
            # horizontal win 
            if (w in range(width - k + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + k))) == 1):
                return True, player
            # vertical win |
            if (h in range(height - k + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + k * width, width))) == 1):
                return True, player
            # up-right diagonal win / 
            if (w in range(width - k + 1) and h in range(height - k + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + k * (width + 1), width + 1))) == 1):
                return True, player
            # down-right diagonal win \
            if (w in range(k - 1, width) and h in range(height - k + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + k * (width - 1), width - 1))) == 1):
                return True, player
            
        return False, -1
    
    def end_game(self):
        has_win, winner = self.has_winner()
        if has_win:
            return True, winner
        elif not len(self.available_moves):
            return True, -1
        
        return False, -1


    def get_current_player(self):
        return self.current_player


class Game(object):
    '''
    Implementing the setup around the game (players, board, displaying)
    '''
    def __init__(self, board):
        self.board = board
    
    def draw_board(self, board, player_1, player_2):
        '''
        draws the board with player and piece info
        '''
        width = board.width
        height = board.height

        print('Player', player_1, 'with ●'.rjust(3))
        print('Player', player_2, 'with ○'.rjust(3))

        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player_1:
                    print('●'.center(8), end='')
                elif p == player_2:
                    print('○'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def play(self, player_1, player_2, start_player=0, display=1):
        if not start_player in (0,1):
            raise Exception("starting player must be 0 (player_1 starting) or 1 (player_2 starting)")
        
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player_1.set_player(p1)
        player_2.set_player(p2)
        player_dict = {p1: player_1, p2: player_2}

        if display:
            self.draw_board(self.board, player_1.player, player_2.player)
        
        while True:
            current_player = self.board.get_current_player()
            turn_player = player_dict[current_player]

            # make a move
            move = turn_player.get_action(self.board)
            self.board.do_move(move)

            # update the display with the new move
            if display:
                self.draw_board(self.board, player_1.player, player_2.player)

            # check if game has ended
            has_end, winner = self.board.end_game()
            if has_end:
                if display:
                    if winner != 0: # some player won
                        print('The game has ended. The winner is', player_dict[winner])
                    else: # tie
                        print('The game has ended. It is a tie.')

                return winner
    
    def self_play(self, player, display=0, temperature=0.01):
        '''
        starts a self-play using MCTS player and reusing the search tree.
        stores the data from self-play as a list of tuples (state, probabilities, winner)
        '''
        self.board.init_board()
        p1, p2 = self.board.players
        
        states, probabilities, current_players = [], [], []
        
        while True:
            # return the move and its probabilities from MCTS
            move, move_probabilities = player.get_action(self.board, temperature, return_probability=1)

            # append the move data to store
            states.append(self.board.current_board())
            probabilities.append(move_probabilities)
            current_players.append(self.board.current_player)

            # perform the move
            self.board.do_move(move)

            # update the display with new move
            if display:
                self.draw_board(self.board, p1, p2)

            # check if game has ended
            has_end, winner = self.board.end_game()
            if has_end:
                winners = np.zeros(len(current_players))
                if winner != 0: # some player won
                    winners[np.array(current_players) == winner] = 1.0
                    winners[np.array(current_players) != winner] = -1.0
                
                # reset the MCTS
                player.reset_player()

                if display:
                    if winner != -1: # some player won
                        print('The game has ended. The winner is player:', winner)
                    else: # tie
                        print('The game has ended. It is a tie.')
                
                return winner, zip(states, probabilities, winners)