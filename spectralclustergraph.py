############################################################################################
# Spectral Clustering Approach
# Author: Nischal Paudyal
# Math2750
# Implementation: Networkx library for creation, manupulation and study of Graph
# Plots: Orginal Graph, Fielder Cluster, K-Mean Cluster,
#        Newmans Modularity Maximization
############################################################################################
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as cvq

import utils as ut
import community_newman as cn

G = nx.Graph()
print("Reading in the Data...\n")
#G = nx.karate_club_graph()

G = nx.read_edgelist("stack_network_nodes.txt", nodetype = str)

print(nx.info(G))
density = nx.density(G)
print("Network density:", density)
diameter = nx.diameter(G)
print("Network Diameter: ", diameter)
al_connectivity = nx.algebraic_connectivity(G)
print("algebric connectivity: ", al_connectivity)

pos = nx.spring_layout(G)

#adding Attributes to the graph
print("pos ", pos)
offsetx = -0.02
offsety = -0.03
pos_labels = {}
keys = pos.keys()
print("Keys", keys)
for key in keys:
    x, y = pos[key]
    pos_labels[key] = (x+offsetx, y+offsety)

print("Pos Label", pos_labels)
fig = plt.figure(1, figsize=(12, 12))
nx.draw_networkx_edges(G, pos, width=1.1, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='aqua')
nx.draw_networkx_labels(G, pos_labels, font_size= 9)
plt.show()

print(nx.info(G))
A = nx.adjacency_matrix(G)

# calculate leoplacian matrix
D = np.diag(np.ravel(np.sum(A, axis=1)))
L = D - A
# print(L)

#eigenvalue and eigenvector of Laplacian Matrix
eigval, eigvec = np.linalg.eigh(L)
roundeVal = np.around(eigval, 3)
#print(b)

#eigenvector corresponding to second smalled eigenvalue
pVec = eigvec[:, 1]
#print(f)

flabels = np.ravel(np.sign(pVec))
#print(labels)

#fiedler cluster plot
fig = plt.figure(2, figsize=(12, 12))
nx.draw_networkx_edges(G, pos, width=1.1, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=flabels)
nx.draw_networkx_labels(G, pos_labels, font_size= 9)
plt.show()

#k-mean calculation
k = 3
means, klabels = cvq.kmeans2(eigvec[:, 1:k], k)
#print(labels2)

#k-mean cluster plot
fig = plt.figure(3, figsize=(12, 12))
nx.draw_networkx_edges(G, pos, width=1.1, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=klabels)
nx.draw_networkx_labels(G, pos_labels, font_size= 9)
plt.show()


#Modularity Maximization
#Libray: https://github.com/zhiyzuo/python-modularity-maximization/tree/master/modularity_maximization
comm_dict = cn.partition(G)
print(comm_dict)
v = []
for comm in comm_dict:
    v.append(comm_dict[comm])
#print(v)

fig = plt.figure(4, figsize=(12, 12))
nx.draw_networkx_edges(G, pos, width=1.1, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=v)
nx.draw_networkx_labels(G, pos_labels, font_size= 9)
plt.show()
print('Modularity of such partition for karate is %.3f' % ut.get_modularity(G, comm_dict))
print("finished")