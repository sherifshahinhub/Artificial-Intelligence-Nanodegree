from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, breadth_first_search, astar_search, depth_first_graph_search,
    uniform_cost_search, greedy_best_first_graph_search, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state
)
from my_planning_graph import PlanningGraph
from run_search import run_search

from functools import lru_cache

class HaveCakeProblem(Problem):
    def __init__(self, initial: FluentState, goal: list):
        self.state_map = initial.pos + initial.neg
        Problem.__init__(self, encode_state(initial, self.state_map), goal=goal)
        self.actions_list = self.get_actions()
    
    def get_actions(self):
        precond_pos = [expr("Have(Cake)")]
        precond_neg = []
        effect_add = [expr("Eaten(Cake)")]
        effect_rem = [expr("Have(Cake)")]
        eat_action = Action(expr("Eat(Cake)"), [precond_pos, precond_neg], [effect_add, effect_rem])
        precond_pos = []
        precond_neg = [expr("Have(Cake)")]
        effect_add = [expr("Have(Cake)")]
        effect_rem = []
        bake_action = Action(expr("Bake(Cake)"), [precond_pos, precond_neg], [effect_add, effect_rem])
        return [eat_action, bake_action]


def have_cake():
    def get_init():
        pos = [expr("Have(Cake)")]
        neg = [expr("Eaten(Cake)")]
        return FluentState(pos, neg)
    def get_goal():
        return [expr("Have(Cake)"), expr("Eaten(Cake)"), ]
    













































































