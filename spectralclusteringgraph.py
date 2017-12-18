import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import scipy.cluster.vq as cvq

import utils as ut
import community_newman as cn


print("Reading in the Data...\n")
G = nx.read_edgelist("email-Eu-core.txt",create_using=nx.DiGraph(), nodetype = int)
#G = nx.powerlaw_cluster_graph(100, 1, 0.1)
#G = nx.karate_club_graph()

#############
print(nx.info(G))
density = nx.density(G)
print("Network density:", density)
#diameter = nx.diameter(G)
#print("Network Diameter: ", diameter)
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)
degree_dict = G.degree(G.nodes())
# nx.set_node_attributes(G, 'degree', degree_dict)
#print(G.node[5])
#betweenness_dict = nx.betweenness_centrality(G)  # Run betweenness centrality
#eigenvector_dict = nx.eigenvector_centrality(G)  # Run eigenvector centrality
#############

############
pos = nx.spring_layout(G)
##nx.draw(G, pos=graphviz_layout(G), node_size=1600, cmap=plt.cm.Blues,
# node_color=range(len(G)),
# prog='dot')
nx.draw_networkx_edges(G, pos)

nx.draw_networkx_nodes(G, pos, node_size=45, node_color='R')
plt.show()


print(nx.info(G))
A = nx.adjacency_matrix(G)

# calculate leoplacian matrix
D = np.diag(np.ravel(np.sum(A, axis=1)))
L = D - A
# print(L)

eigval, eigvec = np.linalg.eigh(L)
b = np.around(eigval, 3)
#print(b)

f = eigvec[:, 1]

# print(f)

labels = np.ravel(np.sign(f))

# print(labels)

fig = plt.figure(3, figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=45, node_color=labels)
nx.draw_networkx_edges(G, pos)
plt.show()

k = 5
means, labels2 = cvq.kmeans2(eigvec[:, 1:k], k)
print(labels2)
nx.draw_networkx_nodes(G, pos, node_size=45, node_color=labels2)
nx.draw_networkx_edges(G, pos)
plt.show()


#comm_dict = cn.partition(G)
#print(comm_dict)
#v = []
#for comm in comm_dict:
#    v.append(comm_dict[comm])
#print(v)
#fig = plt.figure(4, figsize=(12, 12))
#nx.draw_networkx_nodes(G, pos, node_size=45, node_color=v)
#nx.draw_networkx_edges(G, pos)
#plt.show()
#print('Modularity of such partition for karate is %.3f' % ut.get_modularity(G, comm_dict))
print("finished")