import networkx as nx
import numpy as np
import math
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

n = 10
two_pow_13 =pow(2, 13)
two_pow_12 =pow(2, 12)
two_pow_11 =pow(2, 11)
two_pow_10 =pow(2, 10)
two_pow_9 =pow(2, 9)


#graphs
def normalize_mat(adjacency):
    # Calculate row sums
    row_sums = np.sum(adjacency, axis=1)

    # Create diagonal matrix with inverse of row sums
    D = np.diag(1 / np.sqrt(row_sums))

    # Normalize adjacency matrix
    normalized_adj_matrix = D.dot(adjacency)

    return normalized_adj_matrix

def clique(n):
    G = nx.Graph()
    for i in range (1, n+1):
        G.add_node(i)
        G.add_edge(i,i)

    for i in range(1, n+1):
        for j in range(1, n+1):
            if(i < j):
                G.add_edge(i, j)
    adjacency = nx.to_numpy_array(G)
    return G, adjacency

cliqueG, adjacency_mat_clique= clique(n )
norm_mat_clique = normalize_mat(adjacency_mat_clique)
print("finished creating clique")


#fig, ax = plt.subplots()
#nx.draw(cliqueG, with_labels=True, ax=ax)
#ax.set_title("Clique Graph")

#plt.savefig("clique_graph.png")
'''
def tree(n):
    G = nx.Graph()
    for i in range (1, n+1):
        G.add_node(i)
        G.add_edge(i, i)
    for i in range(1, (n//2)+1):
        if 2*i <= n:
            G.add_edge(i, 2*i)
        if 2*i + 1 <= n:
            G.add_edge(i, 2*i + 1)
    return G

treeG = tree(n)
fig, ax = plt.subplots()
nx.draw(treeG, with_labels=True, ax=ax)
ax.set_title("Tree")

plt.savefig("tree_graph.png")
'''

def ring(n):
    G = nx.Graph()
    for i in range (1, n+1):
        G.add_node(i)
        G.add_edge(i,i)
    for i in range (1, n+1):
        if i ==  n:
            G.add_edge(n,1)
        else:
            G.add_edge(i, i+1)
    adjacency = nx.to_numpy_array(G)
    return G, adjacency

ringG, adjacency_mat_ring = ring(n )
norm_mat_ring = normalize_mat(adjacency_mat_ring)
#print("finished creating ring")

#fig, ax = plt.subplots()
#nx.draw(ringG, with_labels=True, ax=ax)
#ax.set_title("Ring Graph")

#plt.savefig("ring_graph.png")


def candy(n):
    G = nx.Graph()
    # Add the chain vertices
    for i in range(1, (n//2)+1):
        G.add_node(i)
        G.add_edge(i, i)
        if i > 1:
            G.add_edge(i-1, i)

    # Add the clique vertices
    for i in range((n//2)+1, n+1):
        G.add_node(i)
        G.add_edge(i, i)


    # Connect the clique vertices
    for i in range(n//2+1, n+1):
        for j in range((n//2)+1, n+1):
            if i != j:
                G.add_edge(i, j)


    G.add_edge(n//2, (n//2)+1)

    adjacency = nx.to_numpy_array(G)
    return G, adjacency

candyG ,adjacency_mat_candy= candy(n )
norm_mat_candy = normalize_mat(adjacency_mat_candy)
#print("finished creating candy")

#fig, ax = plt.subplots()
#nx.draw(candyG, with_labels=True, ax=ax)
#ax.set_title("Candy Graph")
#plt.savefig("candy_graph.png")

def two_cliques(n):
    G = nx.Graph()
    # Add the chain vertices
    for i in range(1, n+1):
        G.add_node(i)
    for i in range(1, (n//2)+1):
        G.add_edge(i, i)
        if i > 1:
            G.add_edge(i-1, i)

    # Connect the clique vertices
    for i in range((n//2)+1,(3*n//4)+1):
        for j in range((n//2)+1, (3*n//4)+1):
            if i <= j:
                G.add_edge(i, j)



    # Connect the clique vertices
    for i in range((3*n//4) +1, n+1):
        for j in range((3*n//4) +1, n+1):
            if i <= j:
                G.add_edge(i, j)


    G.add_edge(1, (n//2)+1)
    G.add_edge(n//2, (3*n//4) +1)

    adjacency = nx.to_numpy_array(G)
    return G, adjacency

two_cliques_G , adjacency_mat_two_cliques= two_cliques(n )
norm_mat_two_cliques = normalize_mat(adjacency_mat_two_cliques)
#print("finished creating two cliques")

#fig, ax = plt.subplots()
#nx.draw(two_cliques_G, with_labels=True, ax=ax)
#ax.set_title("Two Cliques Graph")
#plt.savefig("two_cliques_graph.png")


def stationary_distribution(mat):
    p = []
    #adjacency = nx.to_numpy_array(G)
    for row in mat:
        p.append(sum(row))

    p = np.array(p)
    return p / p.sum()#, adjacency

def normalize_mat(adjacency):
    # Calculate row sums
    row_sums = np.sum(adjacency, axis=1)

    # Create diagonal matrix with inverse of row sums
    D = np.diag(1 / np.sqrt(row_sums))

    # Normalize adjacency matrix
    normalized_adj_matrix = D.dot(adjacency)

    return normalized_adj_matrix

def stationary_distribution(mat):
    p = []
    #adjacency = nx.to_numpy_array(G)
    for row in mat:
        p.append(sum(row))

    p = np.array(p)
    return p / p.sum()#, adjacency

norm_mat_clique = normalize_mat(adjacency_mat_clique)
stationary_clique = stationary_distribution(norm_mat_clique)
#print("stationary clique:", stationary_clique)

norm_mat_ring = normalize_mat(adjacency_mat_ring)
stationary_ring = stationary_distribution(norm_mat_ring)
#print("stationary ring:", stationary_ring)

norm_mat_candy = normalize_mat(adjacency_mat_candy)
stationary_candy = stationary_distribution(norm_mat_candy)
print("stationary candy:", stationary_candy)

norm_mat_two_cliques = normalize_mat(adjacency_mat_two_cliques)
stationary_two_cliques = stationary_distribution(norm_mat_two_cliques)
#print("stationary two cliques:", stationary_two_cliques)

'''
adjacency_mat_clique ,stationary_clique = stationary_distribution(cliqueG)
print("stationary clique:", stationary_clique)


adjacency_mat_ring ,stationary_ring = stationary_distribution(ringG)
print("stationary ring:", stationary_ring)


adjacency_mat_candy ,stationary_candy = stationary_distribution(candyG)
print("stationary candy:", stationary_candy)

adjacency_mat_two_cliques, stationary_two_cliques = stationary_distribution(two_cliques_G)
print("stationary two cliques:", stationary_two_cliques)
'''



def random_walk(G, start_node, N, page_rank_dict, stat_vec, wanted_delta):
    nodes = list(G.nodes())
    visited = list()
    current_node = start_node
    steps = 0
    while len(visited) < N:
        visited.append(current_node)
        steps += 1
        if len(visited) == N-1:
            page_rank_dict[current_node]+= 1
        neighbors = list(G.neighbors(current_node))
        current_node = np.random.choice(neighbors)
    return steps



def histogram(page_rank_dict, trails):
    probability_vec = np.zeros(len(page_rank_dict))
    for key, value in page_rank_dict.items():
        probability_vec[key -1] = value / (trails +1)
    return probability_vec


def rand_walk_loop(G, n, stat_vec, wanted_delta, trials, N):
    # when n is num of nodes in G, stat_vec is the matching vector
    # trails mean t time pick a random node to start the walk from it
    # Calculate the cover time for the uniform initial distribution
    # N is the number of steps in each rand walking
    cover_time = 0
    page_rank_dict= {i: 0 for i in range(1, n+1)}
    for i in range(trials):
        rand_node = np.random.choice(list(G.nodes()))
        cover_time += random_walk(G, rand_node, N, page_rank_dict, stat_vec, wanted_delta)
        probability_vector = histogram(page_rank_dict, i)
        curr_delta =stat_vec- probability_vector
        curr_delta = np.linalg.norm( curr_delta)
        #if i %10 == 0:
            #print(curr_delta)
        if curr_delta <= wanted_delta:
            break
    cover_time /= trials
    #print(probability_vector)
    print(probability_vector)
    print("the current delta is:")
    print(curr_delta)
    print("\n")


    for i in probability_vector:
        print(i )

'''
    print("the sum of the vector is")
    print(sum(probability_vector))
    print("the average  of node in the vector is")
    print(sum(probability_vector) / n)
'''
trials = 10000#try to do this function with while loop and count the trials it takes
#round(n * math.log(n,2))
N = pow(2,6)
wanted_delta = pow(2, -6)
print("page rank starting")
rand_walk_loop(candyG, n, stationary_candy, wanted_delta, trials, N)