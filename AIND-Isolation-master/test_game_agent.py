from sample_players import RandomPlayer, GreedyPlayer
from isolation import Board
from game_agent import IsolationPlayer, MinimaxPlayer, AlphaBetaPlayer
print(AlphaBetaPlayer().alphabeta(Board(RandomPlayer(), GreedyPlayer()), 5))
#print(AlphaBetaPlayer().alphabeta(Board(RandomPlayer(), GreedyPlayer()), 3))