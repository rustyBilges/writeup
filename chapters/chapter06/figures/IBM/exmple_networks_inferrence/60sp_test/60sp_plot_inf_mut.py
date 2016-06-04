import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shandyliaTimme import *
from filter import *
fsa = 20
w_scale = 10000000

dir = "./"

NEW = False 

#jhat = np.genfromtxt("average_jaht_highIR_60sp_mut0.5.csv", delimiter=',') #  zeros((5,5))
if NEW:
	jhat = np.genfromtxt("average_jaht_NEW_60sp_mut0.5.csv", delimiter=',') #  zeros((5,5))
	#jhat = np.genfromtxt("average_jaht_NEW_60sp_mut0.5_constrained.csv", delimiter=',') #  zeros((5,5))
else:
	#jhat = np.genfromtxt("average_jaht_ORIG_60sp_mut0.5.csv", delimiter=',') #  zeros((5,5))
	jhat = np.genfromtxt("average_jaht_ORIG_60sp_mut0.3.csv", delimiter=',') #  zeros((5,5))
#jhat = np.genfromtxt("average_jaht_ORIG_60sp_mut0.8.csv", delimiter=',') #  zeros((5,5))
Nsp = 6
print jhat

G = nx.DiGraph()
Gr = nx.DiGraph()

for sp in range(Nsp):
	G.add_node(sp)
	Gr.add_node(sp)

#G.add_node(5)
pos = []
pos.append((-np.sin(36*np.pi / 180.),0))
pos.append((-np.sin(36*np.pi / 180.) - np.cos(72*np.pi/180.),np.sin(72*np.pi/180.)))
pos.append((np.sin(36*np.pi / 180.),0))
pos.append((np.sin(36*np.pi / 180.) +  np.cos(72*np.pi/180.),np.sin(72*np.pi/180.)))
pos.append((0, 2 * np.cos((36)*np.pi / 180.)))
pos.append((0, 3.5 * np.cos((36)*np.pi / 180.)))

cols = []
cols.append('g')
cols.append('y')
cols.append('g')
cols.append('y')
cols.append('b')
cols.append('r')

antag_edges = []
mut_edges = []
com_edges = []
re_edges = []
rem_edges = []

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
				

#nx.draw_networkx_nodes(G, pos, node_color=cols) #, cmap=plt.get_cmap('jet'), node_color = values)
#nx.draw_networkx_labels(G,pos)

#nx.draw_networkx_edges(G, pos, edgelist=antag_edges, edge_color='k', arrows=True, width=ant_w)
#nx.draw_networkx_edges(G, pos, edgelist=com_edges, edge_color='r', arrows=False, width=com_w)
#nx.draw_networkx_edges(G, pos, edgelist=mut_edges, edge_color='g', arrows=False, width=mut_w)
#plt.xlim([-2,2])
#plt.xlim([-1.25,1.25])
#plt.axes().set_aspect('equal', 'datalim')
#plt.show()

Gr.add_edge(0,1)
Gr.add_edge(2,3)
Gr.add_edge(1,4)
Gr.add_edge(3,4)
re_edges.append((0,1))
re_edges.append((0,3))
re_edges.append((0,4))
re_edges.append((1,4))
re_edges.append((1,5))
re_edges.append((3,4))
re_edges.append((3,5))
re_edges.append((4,5))
re_edges.append((2,1))
re_edges.append((2,4))
if not NEW:
	re_edges.append((0,5))
	re_edges.append((2,5))
rem_edges.append((2,3))
#pos.append((-np.cos(72*np.pi / 180.),0))
#pos.append((-np.cos(72*np.pi / 180.) - np.sin(72*np.pi/180.),np.cos(72*np.pi/180.)))
#pos.append((np.cos(72*np.pi / 180.),0))
#pos.append((np.cos(72*np.pi / 180.) +  np.sin(72*np.pi/180.),np.cos(72*np.pi/180.)))
#pos.append((0, 2* np.sin(72*np.pi / 180.)))
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
nx.draw_networkx_nodes(G, pos, node_color=cols) #, cmap=plt.get_cmap('jet'), node_color = values)
nx.draw_networkx_labels(G,pos)

nx.draw_networkx_edges(G, pos, edgelist=antag_edges, edge_color='k', arrows=True, width=ant_w)
nx.draw_networkx_edges(G, pos, edgelist=com_edges, edge_color='r', arrows=False, width=com_w)
nx.draw_networkx_edges(G, pos, edgelist=mut_edges, edge_color='g', arrows=False, width=mut_w)
#plt.xlim([-2,2])
#plt.xlim([-1.25,1.25])
#plt.axes().set_aspect('equal', 'datalim')
ax = plt.gca()
ax.set_aspect('equal', 'datalim')
ax.annotate("B", (0,0), (0.02,0.95), color='black', fontsize= fsa, fontweight='bold', xycoords='data', textcoords='axes fraction')
plt.axis('off')

plt.subplot(1,2,1)
nx.draw_networkx_nodes(Gr, pos, node_color=cols) #, cmap=plt.get_cmap('jet'), node_color = values)
nx.draw_networkx_edges(Gr, pos, edgelist=re_edges, edge_color='k', arrows=True) #, width=ant_w)
nx.draw_networkx_edges(Gr, pos, edgelist=rem_edges, edge_color='g', arrows=False) #, width=ant_w)
nx.draw_networkx_labels(Gr,pos)
ax = plt.gca()
ax.set_aspect('equal', 'datalim')
ax.annotate("A", (0,0), (0.02,0.95), color='black', fontsize= fsa, fontweight='bold', xycoords='data', textcoords='axes fraction')
plt.axis('off')
if NEW:
	plt.savefig("60species_new_mut0.5_average_Jhat.png")
else:
	#plt.savefig("60species_orig_mut0.5_average_Jhat.png")
	plt.savefig("60species_orig_mut0.3_average_Jhat.png")
plt.show()


