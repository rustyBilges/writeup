
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#graphml_file = "./test/new_network.graphml"
graphml_file = "./network_2619758_126.graphml"
G = nx.read_graphml(graphml_file)
G = nx.Graph(G)
#G=nx.dodecahedral_graph()

#tls = nx.attr_matrix(G, node_attr='tl') #, rc_order=[0,1,2])
tls=nx.get_node_attributes(G,'tl')
print(tls)

ids = range(1,201)


posi = dict()


#for i  in ids:
#	for k in tls.keys():
#		if k==str(i):
#			posi[k] = (ids[i-1]*10000.0,tls[k])
#			#posi[str(i)] = np.asarray([0.5,float(tls[k])/4.0])


node_color = []

# for each node in the graph

px_0 = 1
px_1 = 1
px_2 = 1
px_3 = 1


for node in G.nodes(data=True):
    print(node)
    # if the node has the attribute group1
    if 0 == node[1]['tl']:
        node_color.append('green')
	#posi[node[0]] = (px_0, node[1]['tl'])
	px_0 += 1

    # if the node has the attribute group1
    elif 1 == node[1]['tl']:
        node_color.append('blue')
	#posi[node[0]] = (px_1, node[1]['tl'])
	px_1 += 1

    # if the node has the attribute group1
    elif 2 == node[1]['tl']:
        node_color.append('yellow')
        #fr = int(px_2) / 20
	#posi[node[0]] = (px_2 - (fr*20 -2), node[1]['tl'] + 0.1*fr)
	#posi[node[0]] = (px_2, node[1]['tl'])
	px_2 += 1

    # if the node has the attribute group1
    elif 3 == node[1]['tl']:
        node_color.append('red')
	#posi[node[0]] = (px_3, node[1]['tl'])
	px_3 += 1

node_scaling = 2000
node_sizes = []
sp_0 = 1
sp_1 = 1
sp_2 = 1
sp_3 = 1
for node in G.nodes(data=True):
    if 0 == node[1]['tl']:
        node_sizes.append(node_scaling/px_0)
	posi[node[0]] = (float(sp_0)/float(px_0), node[1]['tl'])
	sp_0 += 1

    elif 1 == node[1]['tl']:
        node_sizes.append(node_scaling/px_1)
	posi[node[0]] = (float(sp_1)/float(px_1), node[1]['tl'])
	sp_1 += 1

    elif 2 == node[1]['tl']:
        node_sizes.append(node_scaling/px_2)
	posi[node[0]] = (float(sp_2)/float(px_2), node[1]['tl'])
	sp_2 += 1

    elif 3 == node[1]['tl']:
        node_sizes.append(node_scaling/px_3)
	posi[node[0]] = (float(sp_3)/float(px_3), node[1]['tl'])
	sp_3 += 1


##for i in range(1,node_count+1):#
#	print G.node[i]['d6']
#posi = nx.spring_layout(G)
#print(posi)
#print(colors)

#nx.draw_networkx(G, pos=posi, node_color=node_color, style='dashed')  
nx.draw_networkx(G, pos=posi, with_labels=False, node_color=node_color, width=0.1, node_size=node_sizes)#, node_shape="|")

ssum = float(px_1 + px_2 + px_3 +px_0)
## try adding legend:
plt.plot([0],[0],color='red',label="TL4: %.3f" %(px_3/ssum))
plt.plot([0],[0],color='yellow',label="TL3: %.3f" %(px_2/ssum))
plt.plot([0],[0],color='blue',label="TL2: %.3f" %(px_1/ssum))
plt.plot([0],[0],color='green',label="TL1: %.3f" %(px_0/ssum))
plt.subplots_adjust(right=0.8)
plt.legend(loc='center right', bbox_to_anchor=(1.2,0.5), ncol=1, fancybox=True, shadow=True)

plt.axis('off')
plt.savefig("example_food_web.png")
plt.show()
