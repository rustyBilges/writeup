
import numpy as np

text1 = '  <key attr.name="tl" attr.type="integer" for="node" id="d6" />\n'
tl_file = "space/2619758_126/output_species.csv"
#tl_file = "predator_over_abundance/3329468_1226/output_species.csv"
tls = np.genfromtxt(tl_file, delimiter=',', skip_header=1)
tls = tls[:,1]

graph_file = "space/2619758_126/initial_network.graphml"
#graph_file = "./test/initial_network.graphml"
new_graph_file = "./network_2619758_126.graphml"
ngf = open(new_graph_file, 'w')

with open(graph_file) as f:
    lines = f.readlines()
    #for line in lines:
    #    pass

sp_id = 1

for line in lines:

	ngf.write(line)


	if line == '  <key attr.name="mut" attr.type="boolean" for="node" id="d0" />\n':
		ngf.write(text1)

	if line == '    <node id="%d">\n' %sp_id:
		ngf.write('      <data key="d6">%d</data>\n' %tls[sp_id-1])
		sp_id += 1
