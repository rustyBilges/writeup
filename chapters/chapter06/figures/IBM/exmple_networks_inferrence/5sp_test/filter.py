import numpy as np

def Filter(pops, wl=100, prepend=False):

	if np.shape(pops)[1] > np.shape(pops)[0]:
		transpose = True
		pops = np.transpose(pops)
	
	length = np.shape(pops)[0]
	new_pops = np.zeros((np.shape(pops)[0]-wl, np.shape(pops)[1]))

	for i in range(length-wl):
		new_pops[i,:] = np.mean(pops[i:i+wl,:],0)

	if prepend:
		preffix = np.zeros((int(wl/2.0), np.shape(new_pops)[1]))
	new_pops = np.concatenate((preffix,new_pops), axis=0)

	if transpose:
		new_pops = np.transpose(new_pops)

	return new_pops
	
