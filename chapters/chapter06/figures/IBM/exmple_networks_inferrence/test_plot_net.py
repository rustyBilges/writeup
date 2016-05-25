import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

T = np.asarray([[0,-1,0],[1,0,-1],[0,1,0]])

#G = nx.Graph(T)
G = nx.DiGraph()

G.add_node(0)
G.add_node(1)
G.add_node(2)

G.add_edge(0,1, weight=1)
G.add_edge(1,2, weight=2)
G.add_edge(2,0, weight=3)


antag_edges = [(0,1), (1,2)]
comp_edges = [(2,0)]

ant_w = [1,2]
com_w = [3]
#edge_colours = ['blue' if not edge in antag_edges else 'red'
#                for edge in G.edges()]


#pos = nx.spring_layout(G)
pos = [(0,0), (0.1,2), (0,4)]
nx.draw_networkx_nodes(G, pos) #, cmap=plt.get_cmap('jet'), node_color = values)
nx.draw_networkx_edges(G, pos, edgelist=antag_edges, edge_color='r', arrows=True, width=ant_w)
nx.draw_networkx_edges(G, pos, edgelist=comp_edges, edge_color='b', arrows=False, width=com_w)
nx.draw_networkx_labels(G,pos)
#nx.draw(G)
plt.show()
