"""
self._board = [[0] * ylim for _ in range(xlim)]
self._board[-1][-1] = 1  # block lower-right corner
self._parity = 0
self._player_locations = [None, None]
"""
"""
def terminal_test(gameState):
    return True if gameState.get_legal_moves() == [] else False


def min_value(gameState):
    if terminal_test(gameState):
        return 1
    v = float('inf')
    for loc in gameState.get_legal_moves():
        v = min(v, max_value(gameState.forecast_move(loc)))
    return v

def max_value(gameState):
    if terminal_test(gameState):
        return -1
    v = float('-inf')
    for loc in gameState.get_legal_moves():
        v = max(v, min_value(gameState.forecast_move(loc)))
    return v

def minimax_decision(gameState):
    best_locations = []
    for location in gameState.get_legal_moves():
        if min_value(gameState.forecast_move(location)) == 1:
            best_locations.append(location)
    return best_locations

"""
###############################################################
###############################################################
###############################################################

# TODO: Add a "depth" parameter to each function
# TODO: Update all recursive calls to use the depth parameter
# TODO: Add a new conditional to cut off search when the depth
#       limit is reached
# NOTE: The minimax_decision function has been done for you!

def minimax_decision(gameState, depth):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    best_score = float("-inf")
    best_move = None
    for m in gameState.get_legal_moves():
        # call has been updated with a depth limit
        v = min_value(gameState.forecast_move(m), depth - 1)
        if v > best_score:
            best_score = v
            best_move = m
    return best_move


def min_value(gameState, depth):
    """ Return the value for a win (+1) if the game is over,
    otherwise return the minimum value over all legal child
    nodes.
    """
    # TODO: add a depth parameter to the function signature.
    if terminal_test(gameState):
        return 1  # by Assumption 2
        
    # TODO: add a new conditional test to cut off search
    #       when the depth parameter reaches 0 -- for now
    #       just return a value of 0 at the depth limit
    #print('.depth: {}'.format(depth))
    if depth <= 0:
        return 0
        
    v = float("inf")
    for m in gameState.get_legal_moves():
        # TODO: pass a decremented depth parameter to each
        #       recursive call
        v = min(v, max_value(gameState.forecast_move(m), depth - 1))
    return v


def max_value(gameState, depth):
    """ Return the value for a loss (-1) if the game is over,
    otherwise return the maximum value over all legal child
    nodes.
    """
    # TODO: add a depth parameter to the function signature.
    if terminal_test(gameState):
        return -1  # by assumption 2
    
    # TODO: add a new conditional test to cut off search
    #       when the depth parameter reaches 0 -- for now
    #       just return a value of 0 at the depth limit
    #print('..depth: {}'.format(depth))
    if depth <= 0:
        return 0
    
    v = float("-inf")
    for m in gameState.get_legal_moves():
        # TODO: pass a decremented depth parameter to each
        #       recursive call
        v = max(v, min_value(gameState.forecast_move(m), depth - 1))
    return v

############################################################
#         DO NOT MODIFY ANYTHING BELOW THIS LINE           #
############################################################

call_counter = 0
def terminal_test(gameState):
    """ Return True if the game is over for the active player
    and False otherwise.
    """
    # NOTE: do NOT modify this function
    global call_counter
    call_counter += 1
    moves_available = bool(gameState.get_legal_moves())  # by Assumption 1
    return not moves_available











































