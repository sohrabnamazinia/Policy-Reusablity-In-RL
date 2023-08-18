import networkx as nx

# Create the directed graph
G = nx.DiGraph()
G.add_edges_from([
    ("b", "a"),
    ("a", "c"),
    ("a", "d"),
    ("c", "e"),
    ("d", "e"),
])

# Specify nodes
a = "a"
b = "b"
c = "c"

# Check if nodes 'b' and 'c' are in the same weakly connected component before removing 'a'

path_exists_before = nx.has_path(G, source=b, target=c)
print(path_exists_before)

# Create a copy of the graph and remove node 'a'
G_copy = G.copy()
G_copy.remove_node(a)

# Check if nodes 'b' and 'c' are in the same weakly connected component after removing 'a'
path_exists_after = nx.has_path(G_copy, source=b, target=c)
print(path_exists_after)
