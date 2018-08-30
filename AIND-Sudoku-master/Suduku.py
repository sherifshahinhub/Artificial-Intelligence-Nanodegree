def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
boxes  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u]) 
             for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s]))
             for s in boxes)


def grid_values(grid):
    dictionary = {}
    for i, val in enumerate(grid):
        if val == '.':
            dictionary[boxes[i]] = '123456789'
        else:
            dictionary[boxes[i]] = val
    return dictionary

def eliminate_notfinished_check(grid):
    for box in grid:
        if len(box) > 1:
            return True
    return False


def eliminate(values):
    #detecting the boxes with one number
    _boxes = []
    for box in boxes:
        if len(values[box]) == 1:
            _boxes.append(box)
    #elemenate        
    for box in _boxes:
        for peer in peers[box]:
            values[peer] = values[peer].replace(values[box],'')
    return values
  
def only_choice(puzzle):
    for unit in unitlist:
        unit_str = ''
        for box in unit:
            unit_str = unit_str + puzzle[box]
        for i in '123456789':
            if unit_str.count(i) == 1:
                for box in unit:
                    if i in puzzle[box]:
                        puzzle[box] = i
    return puzzle
            
def puzzle_solved(puzzle):
    for box in boxes:
        if not len(puzzle[box]) == 1:
            return False
    return True

def constraint_propagation_works(old_puzzle, new_puzzle):
    for box in boxes:
        if old_puzzle[box] != new_puzzle[box]:
            return True
    return False

def no_available_value(puzzle):
    for box in boxes:
        if len(puzzle[box]) == 0:
            return True
    return False

from copy import deepcopy
def reduce_puzzle(puzzle):
    grid = {}
    old_puzzle = {}
    new_puzzle = {}
    if type(puzzle) is str:
        grid = grid_values(puzzle)
        old_puzzle = grid
        new_puzzle = grid
    else:
        old_puzzle = puzzle
        new_puzzle = puzzle
    first = True
    while not puzzle_solved(new_puzzle) and (constraint_propagation_works(old_puzzle, new_puzzle) or first):
        if no_available_value(new_puzzle):
            return False
        first = False
        old_puzzle = deepcopy(new_puzzle)
        new_puzzle = only_choice(eliminate(new_puzzle))
        display(new_puzzle)
    return new_puzzle

def search(puzzle):
    puzzle = reduce_puzzle(puzzle)
    if puzzle is False:
        return False
    if puzzle_solved(puzzle):
        return puzzle
    items_lengths = list([len(f) for f in list(puzzle.values()) if len(f) > 1])
    items = list([f for f in list(puzzle.values()) if len(f) > 1])
    min_box_value = items[items_lengths.index(min(items_lengths))]
    min_box = ''
    for box in puzzle:
        if puzzle[box] == min_box_value:
            min_box = box
            
    for number in puzzle[min_box]:
        try_puzzle = deepcopy(puzzle)
        try_puzzle[min_box] = number
        test_puzzle = search(try_puzzle)
        if test_puzzle:
            return test_puzzle
    
    
def my_naked_twins(puzzle):
    for unit in unitlist:
        for box in unit:
            pointer_1 = box
            if len(puzzle[pointer_1]) == 2:
                for pointer_2 in unit :
                    if pointer_2 != pointer_1 and len(puzzle[pointer_2]) == 2:
                        if puzzle[pointer_1] == puzzle[pointer_2]:
                            for item in unit:
                                if len(puzzle[item]) >= 2 and puzzle[item] != puzzle[pointer_1]:
                                    puzzle[item] = puzzle[item].replace(puzzle[pointer_1][0] ,'')
                                    puzzle[item] = puzzle[item].replace(puzzle[pointer_1][1] ,'')
    return puzzle

def diagonal_constraint():
    unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]+
            [['A1','B2','C3','D4','E5','F6','G7','H8','I9'],[['A9','B8','C7','D6','E5','F4','G3','H2','I1']]]
            )
    print(unitlist)

sudoku = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
hard = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
def main(sudoku_puzzle):
    solved_puzzle = reduce_puzzle(sudoku_puzzle)
    if puzzle_solved(solved_puzzle):
        display(solved_puzzle)
    else:
        display(search(sudoku_puzzle))

def display(values):
    try:
        "Display these values as a 2-D grid."
        width = 1 + max(len(values[s]) for s in boxes)
        line = '+'.join(['-'*(width*3)]*3)
        for r in rows:
            print (''.join(values[r+c].center(width)+('|' if c in '36' else '')
                          for c in cols))
            if r in 'CF': print (line)
        print()
    except:
        print('Error printing the puzzle')
    
















