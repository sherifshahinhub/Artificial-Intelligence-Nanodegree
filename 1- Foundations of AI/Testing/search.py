class search():
    #   '': set([]),
    def __init__(self):
        self.graph = {'Arad': set(['Zerind', 'Sibiu', 'Timasora']),
                      'Zerind': set(['Oradea', 'Arad']),
                      'Sibiu': set(['Fagaras', 'Rimnicu_vilcea', 'Oradea', 'Arad']),
                      'Timasora': set(['Arad', 'Lugoj']),
                      'Oradea': set(['Sibiu', 'Zerind']),
                      'Rimnicu_vilcea': set(['Sibiu', 'Craiova', 'Pitesti']),
                      'Lugoj': set(['Timasora', 'Mehadia']),
                      'Fagaras': set(['Sibiu', 'Bucharest']),
                      'Pitesti': set(['Rimnicu_vilcea', 'Craiova', 'Bucharest']),
                      'Craiova': set(['Rimnicu_vilcea', 'Pitesti', 'Drobeta']),
                      'Mehadia': set(['Lugoj', 'Drobeta']),
                      'Drobeta': set(['Mehadia', 'Craiova']),
                      'Bucharest': set(['Fagaras','Pitesti','Giugiu','Urziceni']),
                      'Giugiu': set(['Bucharest']),
                      'Urziceni': set(['Bucharest'])
                      }
        
    def dfs_paths(self, graph, start, goal, path=None):
        if path is None:
            path = [start]
        if start == goal:
            yield path
        for next in graph[start] - set(path):
            yield from self.dfs_paths(graph, next, goal, path + [next])
            
    def dfs_paths1(self, graph, start, goal):
        stack = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next in graph[vertex] - set(path):
                if next == goal:
                    yield path + [next]
                else:
                    stack.append((next, path + [next]))
  
    def dfs(self, graph, start):
        visited, stack = set(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                stack.extend(graph[vertex] - visited)
        return visited
            
    def dfs1(self, graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for next in graph[start] - visited:
            self.dfs1(graph, next, visited)
        return visited
    
    def bfs(self, graph, start):
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(graph[vertex] - visited)
        return visited
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        