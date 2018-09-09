from copy import deepcopy

xlim, ylim = 3, 2  # board dimensions

class GameState:
    """
    Attributes
    ----------
    _board: list(list)
        Represent the board with a 2d array _board[x][y]
        where open spaces are 0 and closed spaces are 1
    
    _parity: bool
        Keep track of active player initiative (which
        player has control to move) where 0 indicates that
        player one has initiative and 1 indicates player 2
    
    _player_locations: list(tuple)
        Keep track of the current location of each player
        on the board where position is encoded by the
        board indices of their last move, e.g., [(0, 0), (1, 0)]
        means player 1 is at (0, 0) and player 2 is at (1, 0)
    
    """

    def __init__(self):
        self._board = [[0] * ylim for _ in range(xlim)]
        self._board[-1][-1] = 1  # block lower-right corner
        self._parity = 0
        self._player_locations = [None, None]
    
    def get_player_locations(self):
        return self._player_locations
    
    def get_board(self):
        return self._board
    
    def get_parity(self):
        return self._parity
    
    def forecast_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state.
        
        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
        """
        #print(self.get_legal_moves())
        if move not in self.get_legal_moves():
            raise RuntimeError("Attempted forecast of illegal move")
        newBoard = deepcopy(self)
        newBoard._board[move[0]][move[1]] = 1
        newBoard._player_locations[self._parity] = move
        newBoard._parity ^= 1
        return newBoard

    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player. Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.)
        """
        loc = self._player_locations[self._parity]
        if not loc:
            return self._get_blank_spaces()
        moves = []
        rays = [(1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in rays:
            _x, _y = loc
            while 0 <= _x + dx < xlim and 0 <= _y + dy < ylim:
                _x, _y = _x + dx, _y + dy
                if self._board[_x][_y]:
                    break
                moves.append((_x, _y))
        return moves

    def _get_blank_spaces(self):
        """ Return a list of blank spaces on the board."""
        return [(x, y) for y in range(ylim) for x in range(xlim)
                if self._board[x][y] == 0]
        
    
   
    def my_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state.
        
        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
        """
        legal_moves = self.my_get_legal_moves()
        print(legal_moves)
        if move not in legal_moves:
            raise RuntimeError("Attempted forecast of illegal move")
        newBoard = deepcopy(self)
        newBoard._board[move[0]][move[1]] = 1
        newBoard._player_locations[self._parity] = move
        newBoard._parity ^= 1
        return newBoard
    
    
    """
    self._board = [[0] * ylim for _ in range(xlim)]
    self._board[-1][-1] = 1  # block lower-right corner
    self._parity = 0
    self._player_locations = [None, None]
    """
    
    """
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    [(0, 0), (1, 0), (2, 0), (1, 1)]
    [(1, 1), (1, 0), (0, 0)]
    [(1, 0), (1, 1)]
    [(1, 1)]
    """
    def my_get_legal_moves(self):
        location = self._player_locations[self._parity]
        #print('location: {}'.format(location))
        if not location:
            print('First Move')
            return self._get_blank_spaces()
        current_x, current_y = location
        #print('current_x: {0}, current_y: {1}'.format(current_x, current_y))
        #if not current_x and not current_y:
        legal_moves = []
        
        possible_move_directions = [(1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1)]
        for direction in possible_move_directions:
            dir_x, dir_y = direction
            #print('dir_x: {0}, dir_y: {1}'.format(dir_x, dir_y))
            move_x, move_y = current_x + dir_x, current_y + dir_y
            #print('move_x: {0}, move_y: {1}'.format(move_x, move_y))
            while True:
                if 0 <= move_x < xlim and 0 <= move_y < ylim: 
                    if not self._board[move_x][move_y]:
                        #print('move_x: {0}, move_y: {1}'.format(move_x, move_y))
                        legal_moves.append((move_x, move_y))
                        move_x, move_y = move_x + dir_x, move_x + dir_y
                    else: 
                        break
                else:
                    break
        return legal_moves
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
