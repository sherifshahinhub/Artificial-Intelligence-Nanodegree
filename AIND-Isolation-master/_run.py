"""
from isolation import Board
from sample_players import RandomPlayer
from sample_players import GreedyPlayer

greedy_winner = 0
loops = 100
for _ in range(loops):
    # create an isolation board (by default 7x7)
    player1 = RandomPlayer()
    player2 = GreedyPlayer()
    game = Board(player1, player2)
    
    # place player 1 on the board at row 2, column 3, then place player 2 on
    # the board at row 0, column 5; display the resulting board state.  Note
    # that the .apply_move() method changes the calling object in-place.
    game.apply_move((2, 3))
    game.apply_move((0, 5))
    #print(game.to_string())
    
    # players take turns moving on the board, so player1 should be next to move
    assert(player1 == game.active_player)
    
    # get a list of the legal moves available to the active player
    #print(game.get_legal_moves())
    
    # get a successor of the current state by making a copy of the board and
    # applying a move. Notice that this does NOT change the calling object
    # (unlike .apply_move()).
    new_game = game.forecast_move((1, 1))
    assert(new_game.to_string() != game.to_string())
    #print("\nOld state:\n{}".format(game.to_string()))
    #print("\nNew state:\n{}".format(new_game.to_string()))
    
    # play the remainder of the game automatically -- outcome can be "illegal
    # move", "timeout", or "forfeit"
    winner, history, outcome = game.play()
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    #print(game.to_string())
    #print("Move history:\n{!s}".format(history))
    
    if 'Greedy' in str(winner):
        greedy_winner = greedy_winner + 1

print('Greedy won {}% of the time'.format((greedy_winner/loops)*100))
"""
"""
import _playingGame as pg

obj = pg.GameState()
move1 = obj.forecast_move((0,1))
move2 = move1.forecast_move((2,0))
move3 = move2.forecast_move((0,0))
move4 = move3.forecast_move((1,0))
move5 = move4.forecast_move((1,1))
"""
"""
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    [(0, 0), (1, 0), (2, 0), (1, 1)]
    [(1, 1), (1, 0), (0, 0)]
    [(1, 0), (1, 1)]
    [(1, 1)]
"""
"""
obj = pg.GameState()
move1 = obj.my_move((0,1))
move2 = move1.my_move((2,0))
move3 = move2.my_move((0,0))
move4 = move3.my_move((1,0))
move5 = move4.my_move((1,1))
"""
import _playingGame as pg
import _min_max_helpers as h
obj = pg.GameState()
player1 = obj.forecast_move((1,0))
player2 = player1.forecast_move((2,0))
h.minimax_decision(player2, 1)
#Out[68]: (1, 1)
h.minimax_decision(player2, 3)
#Out[67]: (0, 0)
player3 = player2.forecast_move((0,0))
player4 = player3.forecast_move((1,1))
h.minimax_decision(player4, 4)
#Out[73]: (0, 1)
player5 = player4.forecast_move((0,1))
h.minimax_decision(player5, 4)
#In [76]:
player5.get_legal_moves()
#Out[76]: []











