import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

n = pow(2, 14)
two_pow_13 =pow(2, 13)
two_pow_12 =pow(2, 12)
two_pow_11 =pow(2, 11)
two_pow_10 =pow(2, 10)
two_pow_9 =pow(2, 9)

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

cliqueG= clique(two_pow_11)
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
        if i is n:
            G.add_edge(n,1)
        else:
            G.add_edge(i, i+1)
    return G

ringG = ring(two_pow_11)
print("finished creating ring")

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

candyG= candy(two_pow_11)
print("finished creating candy")

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

two_cliques_G = two_cliques(two_pow_11)
print("finished creating two cliques")

#fig, ax = plt.subplots()
#nx.draw(two_cliques_G, with_labels=True, ax=ax)
#ax.set_title("Two Cliques Graph")
#plt.savefig("two_cliques_graph.png")

#question 2
def transition_matrix(G):
    A = nx.to_numpy_array(G)
    return A, A / A.sum(axis=1, keepdims=True)

'''
def stationary_distribution(transition_matrix):
    # Find the eigenvector corresponding to the eigenvalue of 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real
    # Normalize the eigenvector
    return stationary / stationary.sum()'''
def stationary_distribution(transition_matrix):
    w, v = np.linalg.eig(transition_matrix.T)
    i = np.argmin(np.abs(w - 1.0))
    stationary = v[:, i].real
    stationary /= stationary.sum()
    return stationary


adjacency_mat_candy, tran_mat_candy = transition_matrix(candyG)
stationary_candy = stationary_distribution(tran_mat_candy)
print("stationary candy:", stationary_candy)

adjacency_mat_clique,tran_mat_clique = transition_matrix(cliqueG)
stationary_clique = stationary_distribution(tran_mat_clique)
print("stationary clique:", stationary_clique)

adjacency_mat_ring, tran_mat_ring = transition_matrix(ringG)
stationary_ring = stationary_distribution(tran_mat_ring)
print("stationary ring:", stationary_ring)

adjacency_mat_two_cliques, tran_mat_two_cliques = transition_matrix(two_cliques_G)
stationary_two_cliques = stationary_distribution(tran_mat_two_cliques)
print("stationary two cliques:", stationary_two_cliques)


#question 2/2

    

#question 3
'''
def compute_eigenvalue_ratio_with_generalized_power_iteration(adjacency_mat):
    ratios = []
    L = 1/4
    while L >= 1/128:
        # Compute the two largest eigenvalues and eigenvectors of the adjacency matrix
        eigenvalues, eigenvectors = eigsh(adjacency_mat, k=2, which='LM')
        
        # Sort the eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate the ratio of the first and second eigenvalues
        ratio = sorted_eigenvalues[0] / sorted_eigenvalues[1]
        ratios.append(ratio)
        
        # Initialize the eigenvector estimate to a random vector
        adjacency_mat = np.array(adjacency_mat)
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
    
    return ratios

'''
'''
def power_iteration(A, L):
    A=np.array(A)
    n = A.shape[0]
    x0 = np.random.rand(n)
    x0 /= np.linalg.norm(x0)

    for i in range(1000):
        y = A.dot(x0)
        x1 = y / np.linalg.norm(y)
        if np.linalg.norm(x1 - x0) < L:
            break
        x0 = x1

    return x1

def shifted_power_iteration(A, lambda1, L):
    A=np.array(A)
    n = A.shape[0]
    x0 = np.random.rand(n)
    x0 /= np.linalg.norm(x0)

    B = A - lambda1 * np.eye(n)
    B = np.array(B)
    for i in range(1000):
        y = B.dot(x0)
        x2 = y / np.linalg.norm(y)
        if np.linalg.norm(x2 - x0) < L:
            break
        x0 = x2

    return x2

def calculate_eigenvalue_ratio(A, L_values):
    ratios = []
    for L in L_values[:-1]:
        ratios.append("for {}:".format(L))
        lambda1_new = power_iteration(A, L)
        lambda2_new = shifted_power_iteration(A, lambda1_new, L)
        ratios.append(lambda1_new / lambda2_new)

    return ratios

# Example usage
L_values = [1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
'''

def generalized_power_algorithm(G, tol):
    # Initialize the vector x with random values
    x = np.random.rand(len(G))
    # Set the number of iterations and tolerance level
    max_iter = 1000    
    # Run the power iteration
    for i in range(max_iter):
        x_old = x
        x = G @ x
        x = x / np.linalg.norm(x)
        if np.linalg.norm(x - x_old) < tol:
            break
    
    # Calculate the first and second eigenvalues
    lambda1 = np.dot(x, G @ x)
    lambda2 = np.dot(x, G @ G @ x)
    
    # Return the ratio of the first and second eigenvalues
    return lambda1 / lambda2

print("for clique:")
for k in range(2, 7):
    # Compute the tolerance level as 2^-k
    tol = 2 ** -k
    # Compute the ratio of the first and second eigenvalues using the generalized power algorithm
    ratio = generalized_power_algorithm(adjacency_mat_clique, tol)
    # Print the ratio
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", ratio)

print("for ring:")
for k in range(2, 7):
    # Compute the tolerance level as 2^-k
    tol = 2 ** -k
    # Compute the ratio of the first and second eigenvalues using the generalized power algorithm
    ratio = generalized_power_algorithm(adjacency_mat_ring, tol)
    # Print the ratio
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", ratio)

print("for candy:")
for k in range(2, 7):
    # Compute the tolerance level as 2^-k
    tol = 2 ** -k
    # Compute the ratio of the first and second eigenvalues using the generalized power algorithm
    ratio = generalized_power_algorithm(adjacency_mat_candy, tol)
    # Print the ratio
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", ratio)

print("for two cliques:")
for k in range(2, 7):
    # Compute the tolerance level as 2^-k
    tol = 2 ** -k
    # Compute the ratio of the first and second eigenvalues using the generalized power algorithm
    ratio = generalized_power_algorithm(adjacency_mat_two_cliques, tol)
    # Print the ratio
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", ratio)

'''
two_cliques_ratios = calculate_eigenvalue_ratio(adjacency_mat_two_cliques, L_values)
print("two cliques: ")
print(two_cliques_ratios)

candy_ratios = calculate_eigenvalue_ratio(adjacency_mat_candy, L_values)
print("candy: ")
print(candy_ratios)

clique_ratios = calculate_eigenvalue_ratio(adjacency_mat_clique, L_values)
print("clique: ")
print(clique_ratios)

ring_ratios = calculate_eigenvalue_ratio(adjacency_mat_ring, L_values)
print("ring: ")
print(ring_ratios)'''