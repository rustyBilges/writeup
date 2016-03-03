import numpy as np
import matplotlib.pyplot as plt

fsa = 15
results = np.genfromtxt("cov_vs_ir.csv", delimiter=',')

results = results[:,1:]  ## don't trust results for IR=-1e-05

plt.errorbar(results[0,:], results[1,:], yerr=results[2,:], label="HL=0 ; MAI=0.0", fmt='o-')
plt.errorbar(results[0,:], results[3,:], yerr=results[4,:], label="HL=80 ; MAI=0.0", fmt='o-')
plt.errorbar(results[0,:], results[5,:], yerr=results[6,:], label="HL=0 ; MAI=1.0", fmt='o-')
plt.errorbar(results[0,:], results[7,:], yerr=results[8,:], label="HL=80 ; MAI=1.0", fmt='o-')

plt.xlabel("immigration rate (IR)", fontsize=fsa)
plt.ylabel("temporal variability (mean_cv)", fontsize=fsa)

plt.xscale('log')
plt.xlim([0.00008,0.01])
plt.grid()
plt.legend()
#plt.show()
plt.savefig("cov_vs_ir.png")
