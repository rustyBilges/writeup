
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import collections
import os

save_dir = "spatial_anim/"
#subdir = 'testfiles/'
subdir = 'mai0.5_ir0.001_5000/'

pop_file = subdir + "output_pops.csv"
pops = np.genfromtxt(pop_file, delimiter=',',skip_header=1)
species_file = subdir + "output_species.csv"
species = np.genfromtxt(species_file, delimiter=',',skip_header=1)
species2= np.genfromtxt(species_file, delimiter=',',skip_header=1,dtype='S5')

tl = dict(zip(species[:,0], species[:,1]))
ma = dict(zip(species[:,0], species2[:,2]))
mp = dict(zip(species[:,0], species2[:,3]))

files = []
fname = 'animation'
n_frames = 2999

plot_num = 0

fsa = 14
#plt.ion()
fig_hndl = plt.figure(figsize=(20,6.67))

tl0id = []
tl1id = []
tl2id = []
tl3id = []
tl4id = []  ## mutualistic plants
tl5id = []  ## mutualistic animals

for t in tl.keys():
	if tl[t]==0:
		if mp[t]=='False':
			tl0id.append(t-1)  ## -> normal plant
		elif mp[t]=='True':
			tl4id.append(t-1)  ## -> mutualistic plant
	elif tl[t]==1:
		if ma[t]=='False':
			tl1id.append(t-1)  ## -> normal herbivore 
		elif ma[t]=='True':
			tl5id.append(t-1)  ## -> mutualistic animal
	elif tl[t]==2:
		tl2id.append(t-1)
	elif tl[t]==3:
		tl3id.append(t-1)


#in_degrees = G.in_degree()
#out_degrees = G.out_degree()
#tot_degrees = G.degree()
#ext_in_degree = []
#ext_out_degree = []
#ext_tot_degree = []

#norm_in_degree = []
#norm_out_degree = []
##norm_tot_degree = []

#extinctions = np.zeros(4)

def plot_frame(plot_num, tl, pops):

	smin=0
	smax=60
	## plot the spatial states:
	inh = np.genfromtxt(subdir + 'output_inhabitant_%d.csv' %plot_num, delimiter=',')
	vis = np.genfromtxt(subdir + 'output_visitor_%d.csv' %plot_num, delimiter=',')
	#inh = np.genfromtxt(subdir + 'output_inhabitant_%d.csv' %(plot_num+300), delimiter=',')
	#vis = np.genfromtxt(subdir + 'output_visitor_%d.csv' %(plot_num+300), delimiter=',')

	tl0 = np.zeros((np.shape(inh)))
	tl1 = np.zeros((np.shape(inh)))
	tl2 = np.zeros((np.shape(inh)))
	tl3 = np.zeros((np.shape(inh)))
	tl4 = np.zeros((np.shape(inh))) ## mp
	tl5 = np.zeros((np.shape(inh))) ## ma

	for t in tl.keys():
		if tl[t]==0:
			if (t-1) in tl4id:
				tl4 += inh * (inh==t)	
				tl4 += vis * (vis==t)	
			else:
				tl0 += inh * (inh==t)	
				tl0 += vis * (vis==t)	
		elif tl[t]==1:
			if (t-1) in tl5id:
				tl5 += inh * (inh==t)	
				tl5 += vis * (vis==t)	
			else:
				tl1 += inh * (inh==t)	
				tl1 += vis * (vis==t)	
		elif tl[t]==2:
			tl2 += inh * (inh==t)	
			tl2 += vis * (vis==t)	
		elif tl[t]==3:
			tl3 += inh * (inh==t)	
			tl3 += vis * (vis==t)		

	plt.subplot2grid((2,6), (0,0))	
	plt.pcolor(tl0, vmin=smin, vmax=smax)
	plt.title("plants", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((2,6), (0,1))	
	plt.pcolor(tl4, vmin=smin, vmax=smax)
	plt.title("mutualist plants", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((2,6), (0,2))	
	plt.pcolor(tl1, vmin=smin, vmax=smax)
	plt.title("herbivores", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((2,6), (0,3))	
	plt.pcolor(tl5, vmin=smin, vmax=smax)
	plt.title("mutualist animals", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((2,6), (0,4))	
	plt.pcolor(tl2, vmin=smin, vmax=smax)
	plt.title("omnivores", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((2,6), (0,5))	
	plt.pcolor(tl3, vmin=smin, vmax=smax)
	plt.title("predators", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")

	plt.subplot2grid((2,6), (1,0), colspan=6)  ## this to plot dynamics
	plt.plot(np.sum(pops[0:plot_num,:],1),'k', label='total')
	plt.plot(np.sum(pops[0:plot_num,tl0id],1),'g', label='plants')
	plt.plot(np.sum(pops[0:plot_num,tl4id],1),'g--', label='mutualist plants')
	plt.plot(np.sum(pops[0:plot_num,tl1id],1),'b', label='herbivores')
	plt.plot(np.sum(pops[0:plot_num,tl5id],1),'b--', label='mutualist animals')
	plt.plot(np.sum(pops[0:plot_num,tl2id],1),'y', label='omnivores')
	plt.plot(np.sum(pops[0:plot_num,tl3id],1),'r', label='predators')
	plt.xlim([0,n_frames])
	plt.ylim([0,70000])
	plt.legend()
	plt.grid()
	plt.xlabel('iteration', fontsize=12)
	plt.ylabel('number of individuals', fontsize=fsa)

	plt.tight_layout()
	#plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.1, wspace=None, hspace=0.0)
	name = '_tmp%03d.png'%plot_num
	plt.savefig(name)
	files.append(name)
	#plt.show()
	plt.clf()

#plot_frame(1,tl,pops)

for i in range(n_frames): #np.shape(populations)[0]):
    	#for sp in range(np.shape(populations)[1]):
	print(i)

	plot_frame(i+1,tl, pops)
	#plot_frame(i+1833,tl, pops)
#    	for node in G.nodes(data=True):
#		if populations[i,int(node[0])-1]==0.0: # and sp not in extinctions:
			#print("extinct!"
#			print(node)
#			extinctions[node[1]['tl']] += 1
			#ext_degree.append(degrees[node[0]] / (G.number_of_edges()/float(G.number_of_nodes())))
#			ext_in_degree.append(in_degrees[node[0]])
#			ext_out_degree.append(out_degrees[node[0]])
#			ext_tot_degree.append(tot_degrees[node[0]])

#			norm_in_degree.append(G.in_degree(node[0]) / (G.number_of_edges()/float(G.number_of_nodes())))
#			norm_out_degree.append(G.out_degree(node[0]) / (G.number_of_edges()/float(G.number_of_nodes())))
#			norm_tot_degree.append(G.degree(node[0]) / (G.number_of_edges()/float(G.number_of_nodes())))
#
#			G.remove_node(node[0])
#			del posi[node[0]]
#			del node_color[node[0]]
#			del node_sizes[node[0]]

	##update node sizes
#	new_node_sizes = collections.OrderedDict()
#	for ns in node_sizes.keys():
		#print(ns)
#		new_node_sizes[ns] = node_sizes[ns] * (populations[i,int(ns)-1]/max_pop)
	

#os.system("avconv -r 10 -i _tmp%03d.png -b:v 1000k test.mp4")
#os.system("avconv -r 10 -i _tmp%03d.png -b:v 10000k mai0_ir0_1000it.mp4")
os.system("avconv -r 10 -i _tmp%03d.png -b:v 10000k mai0.5_ir0.001_3000it.mp4")
# cleanup
for fname in files: os.remove(fname)
#nx.draw_networkx(G, pos=posi, with_labels=False, node_color=node_color.values(), width=0.1, node_size=node_sizes.values())#, node_shape="|")
#plt.show()
