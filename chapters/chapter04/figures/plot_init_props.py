import numpy as np
import matplotlib.pyplot as plt

texta = 'x-large'
textb = 'x-large'
fs = 20
data = np.genfromtxt('initial_proportions_by_fg_raw.csv', delimiter=',')


plt.subplot(1,3,1)
# The slices will be ordered and plotted counter-clockwise.
labels = 'Producers', 'Animals', 'Predators', 'Top-predators'
temp_sizes = data[:,0]#[15, 30, 45, 1]
sizes = []
sizes.append(temp_sizes[0])
sizes.append(temp_sizes[2])
sizes.append(temp_sizes[4])
sizes.append(temp_sizes[5])
colors = ['green','yellow', 'blue', 'red']

#explode = (0, 0.1, 0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')

patches, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.

for t in texts:
    t.set_size(texta)
for t in autotexts:
    t.set_size(textb)
#autotexts[0].set_color('y')

plt.axis('equal')
plt.title('MAI = 0.0', size=fs)



plt.subplot(1,3,2)
sizes = data[:,2]#[15, 30, 45, 1]
labels = 'Producers', 'Mutualist-producers', 'Animals', 'Mutualist-animals', 'Predators', 'Top-predators'
colors = ['green', 'yellowgreen','yellow', 'gold', 'blue', 'red']

patches, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
for t in texts:
    t.set_size(texta)
for t in autotexts:
    t.set_size(textb)
#autotexts[0].set_color('y')
plt.axis('equal')
plt.title('MAI = 0.5', size=fs)


plt.subplot(1,3,3)
sizes = data[:,4]#[15, 30, 45, 1]
labels = 'Producers', 'Mutualist-producers', 'Animals', 'Mutualist-animals', 'Predators', 'Top-predators'
colors = ['green', 'yellowgreen','yellow', 'gold', 'blue', 'red']

patches, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
for t in texts:
    t.set_size(texta)
for t in autotexts:
    t.set_size(textb)
#autotexts[0].set_color('y')
plt.axis('equal')
plt.title('MAI = 1.0', size=fs)

#plt.tight_layout()
plt.show()
