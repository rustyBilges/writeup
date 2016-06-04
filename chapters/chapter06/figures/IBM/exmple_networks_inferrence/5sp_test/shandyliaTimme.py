"""
Module for estimating the strength of inter-specific interactions from population dynamics.
Uses the method of Shandylia and Timme (2011) to fit a GLV model to the observed data.
The method works by minimising error in observed gradients.

"""
##TODO introduce error handlings, arg checking, edit docstrings
##TODO: how to deal with extinctions
##TODO: allow specification of > groupings (based on e.g. trophic level, functional equivalence..)
##			       > zeros in Jhat (based prior on knowledge that species do not interact)	

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from filter import Filter

def _testData(length=1000, dt=0.01, alpha=1,beta=-1,gamma=-1,delta=1,beta2=0, delta2=0, x0=2,y0=2, plot=False):

	t = 0
	x = 0
	y = 0
	X = []
	Y = []
	T = []
	for i in range(length):
		T.append(t)
		x = x0 + dt*(alpha * x0 + beta * x0 * y0 + beta2*x0*x0)
		y = y0 + dt*(gamma * y0 + delta * x0 * y0 + delta2*y0*y0)
		X.append(x)
		Y.append(y)
		x0 = x
		y0 = y
		t += dt

	if plot:
		plt.plot(T,X,'g')
		plt.plot(T,Y,'r')
		plt.show()

	return (T,X,Y)


def _interpolate(pops, tvec):
	"""Method uses to linear interpolation to evaluate gradients and estimate abundances at intermediate times: tau

	Args:
		pops (numpy array): dynamics in array of dimension = Niter x Nsp
		tvec (numy array): vector of length Niter containing original times

	Returns:
		tuple: (tau, interp_pops, gradients)
	"""
	Nsp = np.shape(pops)[1]

	tau = (tvec[:-1] + tvec[1:]) /2.0 

	interp_pops = (pops[:-1,:] + pops[1:,:]) / 2.0

	#gradients =  (pops[:-1,:] - pops[1:,:])
	gradients =  (pops[1:,:] - pops[:-1,:])

	for i in range(Nsp):
		#gradients[:,i] /= (tvec[:-1] - tvec[1:]) 
		gradients[:,i] /= (tvec[1:] - tvec[:-1]) 
	
	return (tau, interp_pops, gradients)


def jHat(pops,tvec, immigration=False):
	"""Method does the analytic error minimising to fit GLV, returns interaction matrix, estimated growth and immigration parameters.
	"""
	(tau,X,Xdot) = _interpolate(pops,tvec)

	Nobs = len(tau)
	Nsp = np.shape(X)[1]

	Jhat = np.zeros((Nsp,Nsp))
	gr = np.zeros(Nsp)
	ir = np.zeros(Nsp)
	
	err = []

	for i in range(Nsp):
		Xdoti = Xdot[:,i] 
		#Xdoti = Xdot[:,i] - np.ones(Nobs)
		Xi = X[:,i]
		
		if immigration:
			Gi = np.zeros((Nobs,Nsp+2))
			for j in range(Nsp):
				Gi[:,j] = Xi * X[:,j]
			Gi[:,-2] = Xi
			#Gi[:,-1] = 40000*np.ones(Nobs) - np.sum(X,1)
			Gi[:,-1] = 40000*np.ones(Nobs) - Xi
			#Gi[:,-1] =  1 / np.sum(X,1)
			#Gi[:,-1] =np.ones(Nobs) ## No dependence on abundance (true in limit of very few individuals..)
			
			#if i==0:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - X[:,i] #np.sum(X,1) #0.25*np.ones(Nobs) #
			#else:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - (X[:,1]+X[:,2])

		else:
			Gi = np.zeros((Nobs,Nsp+1))
			for j in range(Nsp):
				Gi[:,j] = Xi * X[:,j]
			Gi[:,-1] = Xi
		#print(np.shape(Gi))
		#print(np.shape(Xi))
		#Gi = np.concatenate((Gi,X[:,i]))

		#Jhati = np.matrixmultiply(np.matrixmultiply(Xi,Gi), np.linalg.inv(np.matrixmultiply(np.transpose(Gi),Gi))) 
		#Xi = np.transpose(Xi)
		#Gi = np.transpose(Gi)
		Jhati = np.dot(np.dot(Xdoti,Gi), np.linalg.inv(np.dot(np.transpose(Gi),Gi))) 
		
		#err.append(np.sum(np.abs(Xdoti - np.dot(Jhati,Gi.T))))    ## think this is working, not sure..
                err.append(np.sum(np.abs(Xdoti - np.dot(Jhati,Gi.T)))/ float(np.mean(np.abs(Xdoti))*len(Xdoti)))

		#print(Jhati)
		Jhat[i,:] = Jhati[:Nsp]
		gr[i] = Jhati[Nsp]
		if immigration:
			ir[i] = Jhati[-1]


	#print("Error:")
	#print(err)

	#print(Jhat)
	return (Jhat,gr,ir, err)



def jHat_threshold(pops,tvec, threshold=100):
	""" This version cuts out points that are below a certain threshold in either species, to try and improve accuracy. 
	"""
	immigration=False
	(tau,X,Xdot) = _interpolate(pops,tvec)

	Nsp = np.shape(X)[1]
	idx = X[:,0]>=threshold 
	## thresholding:
	for sp in range(Nsp-1):
		idx += (X[:,sp+1]>=threshold)
	tau = tau[idx]
	X = X[idx,:]	
	Xdot = Xdot[idx,:]

	Nobs = len(tau)

	Jhat = np.zeros((Nsp,Nsp))
	gr = np.zeros(Nsp)
	ir = np.zeros(Nsp)
	
	err = []

	for i in range(Nsp):
		Xdoti = Xdot[:,i]
		Xi = X[:,i]
		
		if immigration:
			Gi = np.zeros((Nobs,Nsp+2))
			for j in range(Nsp):
				Gi[:,j] = Xi * X[:,j]
			Gi[:,-2] = Xi
			#Gi[:,-1] = 80000*np.ones(Nobs) - np.sum(X,1)
			Gi[:,-1] =  1 / np.sum(X,1)
			#Gi[:,-1] = 0.25*np.ones(Nobs) 
			
			#if i==0:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - X[:,i] #np.sum(X,1) #0.25*np.ones(Nobs) #
			#else:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - (X[:,1]+X[:,2])

		else:
			Gi = np.zeros((Nobs,Nsp+1))
			for j in range(Nsp):
				Gi[:,j] = Xi * X[:,j]
			Gi[:,-1] = Xi
		#print(np.shape(Gi))
		#print(np.shape(Xi))
		#Gi = np.concatenate((Gi,X[:,i]))

		#Jhati = np.matrixmultiply(np.matrixmultiply(Xi,Gi), np.linalg.inv(np.matrixmultiply(np.transpose(Gi),Gi))) 
		#Xi = np.transpose(Xi)
		#Gi = np.transpose(Gi)
		Jhati = np.dot(np.dot(Xdoti,Gi), np.linalg.inv(np.dot(np.transpose(Gi),Gi))) 
		
		err.append(np.sum(np.abs(Xdoti - np.dot(Jhati,Gi.T))))    ## think this is working, not sure..

		#print(Jhati)
		Jhat[i,:] = Jhati[:Nsp]
		gr[i] = Jhati[Nsp]
		if immigration:
			ir[i] = Jhati[-1]


	#print("Error:")
	#print(err)

	#print(Jhat)
	return (Jhat,gr,ir, err)

def jHat_impose_topology(pops,tvec, zeros, immigration=False):
	"""Method does the analytic error minimising to fit GLV, returns interaction matrix, estimated growth and immigration parameters.i

		This version can impose zeros in the inferred interaction matrix - does this improve the fit? (based on error function)
	"""
	(tau,X,Xdot) = _interpolate(pops,tvec)

	Nobs = len(tau)
	Nsp = np.shape(X)[1]

	Jhat = np.zeros((Nsp,Nsp))
	gr = np.zeros(Nsp)
	ir = np.zeros(Nsp)
	
	err = []

	for i in range(Nsp):
		Xdoti = Xdot[:,i]
		Xi = X[:,i]
		
		if immigration:
			nsp_reduced = 0  ## reduced number of species for inference based on imposed topology
			indices = []     ## indiced of species included
			for j in range(Nsp):
				if (i,j) not in zeros:
					nsp_reduced += 1
					indices.append(j)
			Gi = np.zeros((Nobs,nsp_reduced+2))
			for j in range(nsp_reduced):
				Gi[:,j] = Xi * X[:,indices[j]]
			Gi[:,-2] = Xi
			#Gi[:,-1] = 40000*np.ones(Nobs) - np.sum(X,1)
			#Gi[:,-1] =np.ones(Nobs) ## No dependence on abundance (true in limit of very few individuals..)
			Gi[:,-1] = 40000*np.ones(Nobs) - Xi
			#if i==0:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - X[:,i] #np.sum(X,1) #0.25*np.ones(Nobs) #
			#else:
			#	Gi[:,-1] = 40000*np.ones(Nobs) - (X[:,1]+X[:,2])

		else:
			nsp_reduced = 0  ## reduced number of species for inference based on imposed topology
			indices = []     ## indiced of species included
			for j in range(Nsp):
				if (i,j) not in zeros:
					nsp_reduced += 1
					indices.append(j)
			Gi = np.zeros((Nobs,nsp_reduced+1))
			for j in range(nsp_reduced):
				Gi[:,j] = Xi * X[:,indices[j]]

			Gi[:,-1] = Xi
		#print(np.shape(Gi))
		#print(np.shape(Xi))
		#Gi = np.concatenate((Gi,X[:,i]))

		#Jhati = np.matrixmultiply(np.matrixmultiply(Xi,Gi), np.linalg.inv(np.matrixmultiply(np.transpose(Gi),Gi))) 
		#Xi = np.transpose(Xi)
		#Gi = np.transpose(Gi)
		Jhati = np.dot(np.dot(Xdoti,Gi), np.linalg.inv(np.dot(np.transpose(Gi),Gi))) 
		#err.append(np.sum(np.abs(Xdoti - np.dot(Jhati,Gi.T))))    ## think this is working, not sure..
                err.append(np.sum(np.abs(Xdoti - np.dot(Jhati,Gi.T)))/ float(np.mean(np.abs(Xdoti))*len(Xdoti)))

		## put zeros back in for removed species:
		jhat_temp = []
		jj = 0
		for jh in Jhati:
			while (i,jj) in zeros:
				jhat_temp.append(0)
				jj += 1
			jhat_temp.append(jh)
			jj += 1
		Jhati = np.asarray(jhat_temp)

		Jhat[i,:] = Jhati[:Nsp]
		gr[i] = Jhati[Nsp]
		if immigration:
			ir[i] = Jhati[-1]


	#print("Error:")
	#print(err)

	#print(Jhat)
	return (Jhat,gr,ir, err)

if __name__=='__main__':
	#T,X,Y = _testData()
	#pops = np.transpose(np.asarray([X,Y]))
	#tvec = np.asarray(T)

	st = 10000
	#st = 3000
	rep='T1'
	pops = np.genfromtxt(rep + "/output_pops.csv", delimiter=',')
	pops = pops[st:,:]

	pops = Filter(pops, 100)  ## new : test it!
	#pops = pops[st:,0:2] ## test first two species only
	#pops = pops[st:,2:] ## test first two species only
	
	#pops = np.transpose(np.asarray([[1,2,3],[1,3,5]]))
	print(np.shape(pops))

	Nsp = np.shape(pops)[1]
	Niter = np.shape(pops)[0]

	## here the series would be sampled if desired. For now we use all time points, for best results possible.
	tvec = np.arange(Niter)
	jhat, gr, it, err = jHat(pops,tvec)

	## now with correct topology imposed:
	#zeros = [(0,1),(0,2), (1,2), (1,3), (2,0), (2,1), (3,0), (3,1)]  ## for testing rep 'B'
	#jhat, gr, it = jHat_impose_topology(pops,tvec,zeros)
	#jhat, gr, ir = jHat(pops,tvec, True)
	#zeros = [(0,1),(1,0), (2,0), (2,1)]  ## for testing 'E'
	#zeros = [(1,2), (2,1)]  ## for testing 'E'
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)

	## for sequential testing of 'A' or T1
	zeros = [(0,0), (1,1)]
	jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)
	
	## for sequential testing of 'B' or T2
	#zeros = [(0,0), (1,1), (2,2), (3,3)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)
	
	#zeros = [(0,0), (1,1), (2,2), (3,3),(0,2),(2,0)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)

	#zeros = [(0,0), (1,1), (2,2), (3,3),(0,2),(2,0), (1,2),(2,1)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)
	
	#zeros = [(0,0), (1,1), (2,2), (3,3),(0,2),(2,0), (1,2),(2,1), (0,3),(3,0)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)
	
	#zeros = [(0,0), (1,1), (2,2), (3,3),(0,2),(2,0) ,(1,2),(2,1), (0,3),(3,0), (1,3),(3,1)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)
	
	#zeros = [(0,0), (1,1), (2,2), (3,3),(0,2),(2,0) ,(1,2),(2,1), (0,3),(3,0), (1,3),(3,1), (0,1), (1,0),(2,3), (3,2)]
	#jhat, gr, it, err = jHat_impose_topology(pops,tvec,zeros)

	plot=False
	if plot==True:
		G = nx.DiGraph(jhat) #nx.from_numpy_matrix(jhat, DiGraph)
		edges = G.edges()
		weights = [G[u][v]['weight'] for u,v in edges]
		weights = np.abs(weights)*(10**5)
		#nx.draw_networkx(G, width=weights)
		nx.draw_networkx(G)
		plt.show()
		#(t,i,g) = _interpolate(pops, tvec)
		#print(t)	
		#print(i)	
		#print(g)	
	plot_dyn = False
	if plot_dyn==True:
		T,X,Y = _testData(7001,1,gr[0], jhat[0,1], gr[1], jhat[1,0], jhat[0,0], jhat[1,1], pops[0,0], pops[0,1])
		#T,X,Y = _testData(1000,0.01,gr[0], jhat[0,1], gr[1], jhat[1,0], jhat[0,0], jhat[1,1], pops[0,0], pops[0,1])

		## plot to compare fitted to original dynamics:
		plt.plot(T,X,'g')
		plt.plot(T,Y,'r')
		plt.plot(tvec,pops[:,0],'b')
		plt.plot(tvec,pops[:,1],'y')
		plt.show()
