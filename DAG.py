import networkx as nx
import env.gridworld

class DAG:
    def __init__(self, n):
        self.graph = nx.DiGraph()
        states = range(n)
        self.graph.add_nodes_from(states)

    def add_edge(self, a, b):
        self.graph.add_edge(a, b)

    # this function is only used for grid world environment
    # and is to convert a state index to its position on the grid
    def index_to_state(self, index, grid_length):
        result = int(index / grid_length), int(index % grid_length)
        return result
    
    # ENV width & length are only used for gridworld policy 
    # to have a better understanding of the position of states 
    def print(self, mode=1, env_length=None):
        print(self.graph)
        if mode == 1:
            n = self.graph.number_of_nodes()
            for i in range(n):
                print("node " + str(i) + ":")
                print("\t" + str(list(self.graph.neighbors(i))))
        elif mode == 2:
            print(self.graph.edges)
        elif mode == 3:
            n = self.graph.number_of_nodes()
            for i in range(n):
                print("node " + str(self.index_to_state(i, env_length)) + ":")
                neighbor_states = [self.index_to_state(neighbor, env_length) for neighbor in self.graph.neighbors(i)]
                print("\t" + str(neighbor_states))
    
    def union(self, other):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.graph.nodes)
        graph.add_edges_from(self.graph.edges)
        graph.add_edges_from(other.graph.edges)
        dag = DAG(self.graph.number_of_nodes())
        dag.graph = graph
        return dag
