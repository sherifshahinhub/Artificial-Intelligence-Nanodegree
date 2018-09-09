import playingGame as pg

obj = pg.GameState()
move1 = obj.forecast_move((0,1))
move2 = move1.forecast_move((2,0))
move3 = move2.forecast_move((0,0))
move4 = move3.forecast_move((1,0))
move5 = move4.forecast_move((1,1))

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


