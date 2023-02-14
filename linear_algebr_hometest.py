import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

n = pow(2, 14)
two_pow_13 =pow(2, 13)
two_pow_12 =pow(2, 12)
two_pow_11 =pow(2, 11)

def clique(n):
    G = nx.Graph()
    for i in range (1, n+1):
        G.add_node(i)
        G.add_edge(i,i)

    for i in range(1, n+1):
        for j in range(1, n+1):
            if(i < j):
                G.add_edge(i, j)
    return G

cliqueG = clique(14)

#fig, ax = plt.subplots()
#nx.draw(cliqueG, with_labels=True, ax=ax)
#ax.set_title("Clique Graph")

#plt.savefig("clique_graph.png")

def ring(n):
    G = nx.Graph()
    for i in range (1, n+1):
        G.add_node(i)
        G.add_edge(i,i)
    for i in range (1, n+1):
        if i is n:
            G.add_edge(n,1)
        else:
            G.add_edge(i, i+1)
    return G

ringG = ring(7)

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
    return G

candyG= candy(7)

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
            if i < j:
                G.add_edge(i, j)


    # Connect the clique vertices
    for i in range((3*n//4) +1, n+1):
        for j in range((3*n//4) +1, n+1):
            if i < j:
                G.add_edge(i, j)
    

    G.add_edge(1, (n//2)+1)
    G.add_edge(n//2, (3*n//4) +1)
    

    return G

two_cliques_G = two_cliques(16)
#fig, ax = plt.subplots()
#nx.draw(two_cliques_G, with_labels=True, ax=ax)
#ax.set_title("Two Cliques Graph")
#plt.savefig("two_cliques_graph.png")
'''
#cover time and random walk code:
 #Performs a random walk on the graph and returns the number of steps it takes to visit all nodes.
def random_walk(G, start_node):
    nodes = list(G.nodes())
    visited = set()
    current_node = start_node
    steps = 0
    while len(visited) < len(nodes):
        steps += 1
        visited.add(current_node)
        neighbors = list(G.neighbors(current_node))
        current_node = np.random.choice(neighbors)
    return steps

def rand_walk_loop(G):
    # Number of trials to perform -???????????????????/ HOW MANT SHOULD I PUT??????????
    trials = 1
    for i in range(11):
        # Calculate the cover time for the uniform initial distribution
        cover_time = 0
        st = time.time()
        for i in range(trials):
            rand_node = np.random.choice(list(G.nodes()))
            cover_time += random_walk(G, rand_node)
        cover_time /= trials
        end = time.time()
        print("Expected cover time for the uniform initial distribution:", cover_time)
        print('Execution time:', end- st, 'seconds')
        if (end- st) > 600:
            print("more than 10 min \n\n\n\n")
            break





def rand_walk_centered_loop(G, node):
    # Number of trials to perform -???????????????????/ HOW MANT SHOULD I PUT??????????
    trials = 1
    for i in range(11):
        # Calculate the cover time for the initial distribution centered on node 0- NEED TO CHANGE 0 TO SOMETHING ELSE
        cover_time = 0
        st = time.time()  
        for i in range(trials):
            cover_time += random_walk(G, node)
        cover_time /= trials
        print("Expected cover time for the initial distribution centered on node (in clique):", cover_time)
        end = time.time()
        print('Execution time:', end- st, 'seconds')
        if (end- st) > 600:
            print("more than 10 min \n\n\n\n")
            break




print("\n\n")
print("this walk is for centered clique with 2 pow 14\n")
rand_walk_centered_loop(cliqueG, 1)
print("\n\n")
print("this walk is for centered ring with 2 pow 12\n")
rand_walk_centered_loop(ringG, 1)


print("this walk is for candy with 2 pow 11\n")
rand_walk_loop(candyG)
print("\n\n")
'''

#question 2
def transition_matrix(G):
    A = nx.to_numpy_array(G)
    return A, A / A.sum(axis=1, keepdims=True)


def stationary_distribution(transition_matrix):
    # Find the eigenvector corresponding to the eigenvalue of 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real
    # Normalize the eigenvector
    return stationary / stationary.sum()
    
#adjacency_mat_candy = nx.to_numpy_array(candyG)
adjacency_mat_candy, tran_mat_candy = transition_matrix(candyG)
stationary_candy = stationary_distribution(tran_mat_candy)

#adjacency_mat_clique = nx.to_numpy_array(cliqueG)
adjacency_mat_clique, tran_mat_clique = transition_matrix(cliqueG)
stationary_clique = stationary_distribution(tran_mat_clique)

#adjacency_mat_ring = nx.to_numpy_array(ringG)
adjacency_mat_ring, tran_mat_ring = transition_matrix(ringG)
stationary_ring = stationary_distribution(tran_mat_ring)

#adjacency_mat_two_cliques = nx.to_numpy_array(two_cliques_G)
adjacency_mat_two_cliques, tran_mat_two_cliques = transition_matrix(two_cliques_G)
stationary_two_cliques = stationary_distribution(tran_mat_two_cliques)

print("stationary candy:", stationary_candy)

print("stationary ring:", stationary_ring)

print("stationary two cliques:", stationary_two_cliques)#for some reason only 1 value is different

print("stationary clique:", stationary_clique)


#question 3
def compute_eigenvalue_ratio_with_power_iteration(adjacency_mat):
    L = 1/4
    while L >= 1/128:
        # Compute the two largest eigenvalues and eigenvectors of the adjacency matrix
        eigenvalues, eigenvectors = eigsh(adjacency_mat, k=2, which='LM')
        
        # Sort the eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate the ratio of the first and second eigenvalues
        ratio = sorted_eigenvalues[0] / sorted_eigenvalues[1]
        print("Ratio:", ratio)
        
        # Initialize the eigenvector estimate to a random vector
        x = np.random.rand(adjacency_mat.shape[0])
        
        # Iterate until the distance between two consecutive iterations is less than or equal to L
        prev_x = x.copy()
        x = adjacency_mat @ x
        x /= np.linalg.norm(x, ord=2)
        distance = np.linalg.norm(x - prev_x, ord=2)
        while distance > L:
            prev_x = x.copy()
            x = adjacency_mat @ x
            x /= np.linalg.norm(x, ord=2)
            distance = np.linalg.norm(x - prev_x, ord=2)
        
        # Divide L by 2 for the next iteration
        L /= 2





def normalize(v):
    return v / np.linalg.norm(v)


print("two cliques: ")
compute_eigenvalue_ratio_with_power_iteration(adjacency_mat_two_cliques)

print("candy: ")
compute_eigenvalue_ratio_with_power_iteration(adjacency_mat_candy)

print("ring: ")
compute_eigenvalue_ratio_with_power_iteration(adjacency_mat_ring)

print("clique: ")
compute_eigenvalue_ratio_with_power_iteration(adjacency_mat_clique)