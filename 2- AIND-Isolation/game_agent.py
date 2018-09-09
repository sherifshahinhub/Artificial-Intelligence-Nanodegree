"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
best_move_ = None
class SearchTimeout(Exception):
    pass

def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    my_moves = float(len(game.get_legal_moves(player)))
    opp_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    return my_moves - 2*opp_moves


def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    my_moves = float(len(game.get_legal_moves(player)))
    opp_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    return my_moves - 3*opp_moves


def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    my_moves = float(len(game.get_legal_moves(player)))
    opp_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    return my_moves - opp_moves


class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            
            for depth in range(self.search_depth):
                best_move = self.minimax(game, depth+1)

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        best_score = float("-inf")
        best_move = None
        for m in game.get_legal_moves():
            v = self.min_value(game.forecast_move(m), depth - 1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move
    
    def min_value(self, gameState, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()       
                        
        if self.terminal_test(gameState) or depth <= 0:
            return self.score(gameState, gameState._inactive_player)
        v = float("inf")
        for m in gameState.get_legal_moves():
            v = min(v, self.max_value(gameState.forecast_move(m), depth - 1))
        return v

    def max_value(self, gameState, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(gameState) or depth <= 0:
            return self.score(gameState, gameState._active_player)
        
        v = float("-inf")
        for m in gameState.get_legal_moves():
            v = max(v, self.min_value(gameState.forecast_move(m), depth - 1))
        return v
    
    def terminal_test(self, gameState):
        moves_available = bool(gameState.get_legal_moves())
        return not moves_available

class v_class():
    def __init__(self, v, m):
        self._value = v
        self._move = m
        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            very_large_number = 100000
            for depth in range(very_large_number):
                best_move = self.alphabeta(game, depth+1)
        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed
        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        v = self.max_value(game, depth, alpha, beta) 
        return v._move
    
    def min_value(self, gameState, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()    
            
        v = v_class(0., None)
        if self.terminal_test(gameState) or depth <= 0:
            v._value = self.score(gameState, gameState._inactive_player)
            return v
        
        v._value = float("inf")
        for m in gameState.get_legal_moves():
            x = self.max_value(gameState.forecast_move(m), depth - 1, alpha, beta)
            if x._value < v._value:
                v._value = x._value
                v._move = m
            if v._value <= alpha:
                v._move = m
                return v
            beta = min(beta, v._value)
        return v

    def max_value(self, gameState, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        v = v_class(0., None)
        if self.terminal_test(gameState) or depth <= 0:
            v._value = self.score(gameState, gameState._active_player)
            return v
        
        v._value = float("-inf")
        for m in gameState.get_legal_moves():
            x = self.min_value(gameState.forecast_move(m), depth - 1, alpha, beta)
            if x._value > v._value:
                v._value = x._value
                v._move = m
            if v._value >= beta:
                v._move = m
                return v
            alpha = max(alpha, v._value)
        return v
    
    
    def terminal_test(self, gameState):
        moves_available = bool(gameState.get_legal_moves())
        return not moves_available



























