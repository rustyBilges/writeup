
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import collections
import os, sys

subdir = sys.argv[1] + '/'

n_frames = 5000

pop_file = subdir + "output_populations.csv"
pops = np.genfromtxt(pop_file, delimiter=',',skip_header=1)
species_file = subdir + "output_species.csv"
species = np.genfromtxt(species_file, delimiter=',',skip_header=1)
species2= np.genfromtxt(species_file, delimiter=',',skip_header=1,dtype='S5')

tl = dict(zip(species[:,0], species[:,1]))
ma = dict(zip(species[:,0], species2[:,3]))
mp = dict(zip(species[:,0], species2[:,4]))

fsa = 14
#plt.ion()
fig_hndl = plt.figure(figsize=(11.5,18))

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

	plt.subplot2grid((3,2), (0,0))	
	plt.pcolor(tl0, vmin=smin, vmax=smax)
	plt.title("(A) plants", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((3,2), (0,1))	
	plt.pcolor(tl4, vmin=smin, vmax=smax)
	plt.title("(B) mutualist plants", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((3,2), (1,0))	
	plt.pcolor(tl1, vmin=smin, vmax=smax)
	plt.title("(C) herbivores", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((3,2), (1,1))	
	plt.pcolor(tl5, vmin=smin, vmax=smax)
	plt.title("(D) mutualist animals", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((3,2), (2,0))	
	plt.pcolor(tl2, vmin=smin, vmax=smax)
	#plt.title("omnivores", fontsize=fsa)#, verticalalignment='top')
	plt.title("(E) prim. predators", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")
	
	plt.subplot2grid((3,2), (2,1))	
	plt.pcolor(tl3, vmin=smin, vmax=smax)
	plt.title("(F) top predators", fontsize=fsa)#, verticalalignment='top')
	plt.axis("off")

	plt.tight_layout()
	#plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.1, wspace=None, hspace=0.0)
	#name = '_tmp%03d.png'%plot_num
	#plt.savefig(name)
	#files.append(name)
	#plt.show()
	#plt.clf()	
	plt.savefig("spatial_state_mai0.5.png")
	#plt.show()


plot_frame(5000, tl, pops)
