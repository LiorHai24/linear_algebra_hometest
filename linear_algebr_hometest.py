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
    adjacency = nx.to_numpy_array(G)
    return G, adjacency

cliqueG, adjacency_mat_clique= clique(two_pow_9)
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

ringG, adjacency_mat_ring = ring(two_pow_9)
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

    adjacency = nx.to_numpy_array(G)
    return G, adjacency

candyG ,adjacency_mat_candy= candy(two_pow_9)
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

two_cliques_G , adjacency_mat_two_cliques= two_cliques(two_pow_9)
print("finished creating two cliques")

#fig, ax = plt.subplots()
#nx.draw(two_cliques_G, with_labels=True, ax=ax)
#ax.set_title("Two Cliques Graph")
#plt.savefig("two_cliques_graph.png")

#question 2
'''
def stationary_distribution(adjacency):
    p = []
    #adjacency = nx.to_numpy_array(G)
    for row in adjacency:
        p.append(sum(row))

    p = np.array(p)
    return p / p.sum()#, adjacency

stationary_clique = stationary_distribution(adjacency_mat_clique)
print("stationary clique:", stationary_clique)

stationary_ring = stationary_distribution(adjacency_mat_ring)
print("stationary ring:", stationary_ring)

stationary_candy = stationary_distribution(adjacency_mat_candy)
print("stationary candy:", stationary_candy)

stationary_two_cliques = stationary_distribution(adjacency_mat_two_cliques)
print("stationary two cliques:", stationary_two_cliques)'''
#question 2/2
'''
def pagerank(M, num_iterations=100, d=0.85):
    """
    Implements the PageRank algorithm for computing the importance of web pages.
    M: adjacency matrix where M[i][j] is 1 if there is a link from page j to page i, 0 otherwise
    num_iterations: number of iterations to run the algorithm
    d: damping factor
    """
    N = len(M)
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v
    
'''
#question 3
 # Define power iteration method
def power_iteration(A, v):
    Av = np.dot(A,v)
    v_new = Av / np.linalg.norm(Av)
    return v_new

def get_highest_eigenvalue(graph,A, tol):
    # Initialize random vector
    x = np.random.rand(len(graph))
   
    # Calculate adjacency matrix and normalize to obtain transition matrix
    D = np.diag(np.sum(A, axis=1))
   # P = np.linalg.inv(D).dot(A)

    lambda_1_old = 0
    old_x = x
    while True:
        x = power_iteration(A, x)
        a = np.dot(x,A)
        b = np.dot(a, x)
        lambda_1 = b / np.dot(x,x)

        #if abs(lambda_1 - lambda_1_old) < tol:
        if abs(np.linalg.norm(x-old_x)) < tol:
            break
        lambda_1_old = lambda_1
        old_x = x
        
    return lambda_1, x

def projection(u, v):
    # calculate dot product of u and v
        dot_uv = np.dot(u, v)

    # calculate dot product of u with itself
        dot_uu = np.dot(u, u)

    # calculate projection of v onto u
        proj_uv = (dot_uv / dot_uu) * u
        return proj_uv

def get_2nd_highest_eigenvalue(graph,A,v, tol):
    w = np.random.rand(len(graph))

    proj_uv = projection(v, w)
    u = w - proj_uv

    D = np.diag(np.sum(A, axis=1))
    #P = np.linalg.inv(D).dot(A)

    lambda_2_old = 0
    old_u = u
    while True:
        u = power_iteration(A, u)
        a = np.dot(u,A)
        b = np.dot(a, u)
        lambda_2 = b / np.dot(u,u)
        #if abs(lambda_2 - lambda_2_old) < tol:
        if abs(np.linalg.norm(u - old_u)) < tol:
            break
        lambda_2_old = lambda_2
        old_u = u
        
    return lambda_2

def get_2_highest(G, A, tol):
    lambda_1, x= get_highest_eigenvalue(G, A, tol)
    lambda_2= get_2nd_highest_eigenvalue(G, A, x, tol)
    return lambda_1, lambda_2

print("for clique:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(cliqueG, adjacency_mat_clique, tol)
    print(first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)
    
print("for ring:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(ringG, adjacency_mat_ring, tol)
    print(first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)


print("for candy:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(candyG, adjacency_mat_candy, tol)
    print(first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)


print("for two cliques:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(two_cliques_G, adjacency_mat_two_cliques, tol)
    print(first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)
