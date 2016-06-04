import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shandyliaTimme import *
from filter import *

w_scale = 10000000

pops = np.genfromtxt("output_pops.csv", delimiter=',')
sps = np.genfromtxt("output_species.csv", delimiter=',')

pops = pops[-1000:,:]

Nsp = np.shape(pops)[1]
Niter = np.shape(pops)[0]

## here the series would be sampled if desired. For now we use all time points, for best results possible.
tvec = np.arange(Niter)
jhat, gr, it, err = jHat(pops,tvec)

print jhat

G = nx.DiGraph()

for sp in range(Nsp):
	G.add_node(sp)

pos = []
pos.append((-np.sin(36*np.pi / 180.),0))
pos.append((-np.sin(36*np.pi / 180.) - np.cos(72*np.pi/180.),np.sin(72*np.pi/180.)))
pos.append((np.sin(36*np.pi / 180.),0))
pos.append((np.sin(36*np.pi / 180.) +  np.cos(72*np.pi/180.),np.sin(72*np.pi/180.)))
pos.append((0, 2 * np.cos((36)*np.pi / 180.)))

antag_edges = []
mut_edges = []
com_edges = []

ant_w = []
com_w = []
mut_w = []

for i in range(Nsp):
	for j in range(i,Nsp):
		if i!=j:
			if jhat[i,j]>0 and jhat[j,i]<0:
				print "i eats j"
				G.add_edge(j,i,weight= (jhat[i,j]+jhat[j,i])/2.)	
				antag_edges.append((j,i))
				ant_w.append(np.log(w_scale * (np.abs(jhat[i,j])+np.abs(jhat[j,i]))/2.))
			elif jhat[i,j]<0 and jhat[j,i]>0:
				print "j eats i"	
				G.add_edge(i,j,weight= (jhat[i,j]+jhat[j,i])/2.)	
				antag_edges.append((i,j))
				#ant_w.append(np.log(w_scale * (jhat[i,j]+jhat[j,i])/2.))
				ant_w.append(np.log(w_scale * (np.abs(jhat[i,j])+np.abs(jhat[j,i]))/2.))
			elif jhat[i,j]<0 and jhat[j,i]<0:
				print "competition"	
				G.add_edge(j,i,weight= (jhat[i,j]+jhat[j,i])/2.)	
				com_edges.append((j,i))
				#com_w.append(np.log(w_scale * (jhat[i,j]+jhat[j,i])/2.))
				com_w.append(np.log(w_scale * (np.abs(jhat[i,j])+np.abs(jhat[j,i]))/2.))
			elif jhat[i,j]>0 and jhat[j,i]>0:
				print "mutualism"	
				G.add_edge(j,i,weight= (jhat[i,j]+jhat[j,i])/2.)	
				mut_edges.append((j,i))
				#mut_w.append(np.log(w_scale * (jhat[i,j]+jhat[j,i])/2.))
				mut_w.append(np.log(w_scale * (np.abs(jhat[i,j])+np.abs(jhat[j,i]))/2.))
				
#pos.append((-np.cos(72*np.pi / 180.),0))
#pos.append((-np.cos(72*np.pi / 180.) - np.sin(72*np.pi/180.),np.cos(72*np.pi/180.)))
#pos.append((np.cos(72*np.pi / 180.),0))
#pos.append((np.cos(72*np.pi / 180.) +  np.sin(72*np.pi/180.),np.cos(72*np.pi/180.)))
#pos.append((0, 2* np.sin(72*np.pi / 180.)))
nx.draw_networkx_nodes(G, pos) #, cmap=plt.get_cmap('jet'), node_color = values)

nx.draw_networkx_edges(G, pos, edgelist=antag_edges, edge_color='r', arrows=True, width=ant_w)
nx.draw_networkx_edges(G, pos, edgelist=com_edges, edge_color='b', arrows=False, width=com_w)
nx.draw_networkx_edges(G, pos, edgelist=mut_edges, edge_color='g', arrows=False, width=mut_w)
#plt.xlim([-2,2])
plt.xlim([-1.25,1.25])
plt.show()

if False:
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
