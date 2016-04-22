import matplotlib.pyplot as plt
import numpy as np

sim_length = 5000 #50000
yma = 80000
shade=0.1
shadecol= 'blue'

fig, axes = plt.subplots(1,1, figsize=(10,6))
#AX = axes.flatten()
AX = []
#AX.append(axes)
dir = './'
#dir = '../network_7_highIM/'

fsa = 18 

ax0 = plt.subplot2grid((1,3), (0,0), colspan=2)
ax1 = plt.subplot2grid((1,3), (0,2), colspan=1)
AX.append(ax0)
AX.append(ax1)

leg = ['TL1', 'TL2', 'TL3', 'TL4', 'total']

pops = np.genfromtxt(dir + 'output_populations.csv', delimiter=',', skip_header=1)
sps = np.genfromtxt(dir + 'output_species.csv', delimiter=',', skip_header=1)
pops = pops[:,1:]
#pops = np.genfromtxt(dir + 'output_pops.csv', delimiter=',')
#sps = np.genfromtxt(dir + 'output_species.csv', delimiter=',', skip_header=1)
#tls = sps[:,1]
tls = dict()
ii = 0
tlss = sps[:,1]
ids = sps[:,0]
for i in ids:
	tls[i] = tlss[ii]
	ii+=1

sor = sorted(tls.items())
tls = []
ids = []
for s in sor:
	tls.append(s[1])
	ids.append(s[0])
tls = np.asarray(tls)

#tls = sps3[:,1]

print(np.shape(pops))
print(tls==2)
pops0 = np.sum(pops[:,tls==0], 1)
pops1 = np.sum(pops[:,tls==1], 1)
pops2 = np.sum(pops[:,tls==2], 1)
pops3 = np.sum(pops[:,tls==3], 1)

biomass = np.sum(pops,1)

#plt.subplot(1,2,1)
AX[0].plot(pops0, 'g')
AX[0].plot(pops1, 'y')
AX[0].plot(pops2, 'b')
AX[0].plot(pops3, 'r')
AX[0].plot(biomass, 'k')
AX[0].set_xlabel("time", fontsize=fsa)
AX[0].set_ylabel("individuals", fontsize=fsa)
AX[0].set_ylim([0,yma])
AX[0].set_xlim([0,sim_length])
AX[0].legend(leg)
AX[0].annotate("A", (0,0), (0.02,0.95), color='black', fontsize= fsa, fontweight='bold', xycoords='data', textcoords='axes fraction')
AX[0].fill_between([0,1000], 0, yma, alpha = shade)
#AX[0].set_title("Immigration = 0.001")
AX[0].grid()
#dir = 'IM_0.0001/'
ent = 1000
AX[1].plot(pops0[0:ent], 'g')
AX[1].plot(pops1[0:ent], 'y')
AX[1].plot(pops2[0:ent], 'b')
AX[1].plot(pops3[0:ent], 'r')
AX[1].plot(biomass[0:ent], 'k')
AX[1].set_xlabel("time", fontsize=fsa)
AX[1].set_ylabel("individuals", fontsize=fsa)
AX[1].set_ylim([0,yma])
AX[1].set_xlim([0,1000])
AX[1].fill_between([0,1000], 0, yma, alpha = shade)
AX[1].grid()
#AX[1].legend(leg)
AX[1].annotate("B", (0,0), (0.02,0.95), color='black', fontsize= fsa, fontweight='bold', xycoords='data', textcoords='axes fraction')
#AX[1].set_title("transience")


#pops = np.genfromtxt(dir + 'output_pops.csv', delimiter=',')
#pops = np.genfromtxt(dir + 'output_pops2.csv', delimiter=',')
#pops = np.genfromtxt(dir + 'output_pops3.csv', delimiter=',')

#sps = np.genfromtxt(dir + 'output_species.csv', delimiter=',', skip_header=1)
#sps2 = np.genfromtxt(dir + 'output_species2.csv', delimiter=',', skip_header=1)
#sps3 = np.genfromtxt(dir + 'output_species3.csv', delimiter=',', skip_header=1)

#tls = sps[:,1]
#tls = sps2[:,1]
#tls = sps3[:,1]
#
#pops0 = np.sum(pops[:,tls==0], 1)
#pops1 = np.sum(pops[:,tls==1], 1)
#pops2 = np.sum(pops[:,tls==2], 1)
#pops3 = np.sum(pops[:,tls==3], 1)

#plt.subplot(1,2,2)
#AX[1].plot(pops0, 'g')
#AX[1].plot(pops1, 'y')
#AX[1].plot(pops2, 'b')
#AX[1].plot(pops3, 'r')
#AX[1].set_xlabel("iteration")
#AX[1].set_ylim([0,40000])
#AX[1].legend(leg)
#AX[1].set_title("Immigration = 0.0001")

plt.tight_layout()
#plt.savefig("example_trophic_dynamics_default.png")
plt.savefig("example_trophic_dynamics_default_HL40_mai1.png")
plt.show()
                                                                                                                                                               

