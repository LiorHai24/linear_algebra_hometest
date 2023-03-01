import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

n = pow(2, 14)
two_pow_13 =pow(2, 13)
two_pow_12 =pow(2, 12)
two_pow_11 =pow(2, 11)
two_pow_10 =pow(2, 10)
two_pow_9 =pow(2, 9)
'''
def normalize_mat(adjacency):
    # Calculate row sums
    row_sums = np.sum(adjacency, axis=0)

    # Create diagonal matrix with inverse of row sums
    #D = np.diag(1 / row_sums)
    
    for row in adjacency:
        for i in row:
            if i != 0:
                D[row][i] = 1 / row_sums
    # Normalize adjacency matrix
    #normalized_adj_matrix = D.dot(adjacency)

    return D'''
def normalize_mat(matrix):
    new_matrix = []
    for row in matrix:
        count_ones = sum(row)
        if count_ones > 0:
            new_row = [1/count_ones if val == 1 else val for val in row]
        else:
            new_row = row
        new_matrix.append(new_row)
    return new_matrix

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

cliqueG, adjacency_mat_clique= clique(two_pow_10)
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

ringG, adjacency_mat_ring = ring(two_pow_10)
norm_mat_ring = normalize_mat(adjacency_mat_ring)
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

candyG ,adjacency_mat_candy= candy(two_pow_10)
norm_mat_candy = normalize_mat(adjacency_mat_candy)
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

two_cliques_G , adjacency_mat_two_cliques= two_cliques(two_pow_10)
norm_mat_two_cliques = normalize_mat(adjacency_mat_two_cliques)
print("finished creating two cliques")

#fig, ax = plt.subplots()
#nx.draw(two_cliques_G, with_labels=True, ax=ax)
#ax.set_title("Two Cliques Graph")
#plt.savefig("two_cliques_graph.png")

#question 2
'''
def stationary_distribution(mat):
    p = []
    #adjacency = nx.to_numpy_array(G)
    for row in mat:
        p.append(sum(row))

    p = np.array(p)
    return p / p.sum()#, adjacency

norm_mat_clique = normalize_mat(adjacency_mat_clique)
stationary_clique = stationary_distribution(norm_mat_clique)
print("stationary clique:", stationary_clique)

norm_mat_ring = normalize_mat(adjacency_mat_ring)
stationary_ring = stationary_distribution(norm_mat_ring)
print("stationary ring:", stationary_ring)

norm_mat_candy = normalize_mat(adjacency_mat_candy)
stationary_candy = stationary_distribution(norm_mat_candy)
print("stationary candy:", stationary_candy)

norm_mat_two_cliques = normalize_mat(adjacency_mat_two_cliques)
stationary_two_cliques = stationary_distribution(norm_mat_two_cliques)
print("stationary two cliques:", stationary_two_cliques)
#question 2/2
'''
'''
def pagerank(M, num_iterations=100, d = pow(2,-6)):#d = 0.85
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
    
print(pagerank(adjacency_mat_ring))
'''
#question 3

 # Define power iteration method
#in progress
def power_iteration(A, v):
    Av = np.dot(A,v)
    v_new = Av / np.linalg.norm(Av)
    return v_new

def get_highest_eigenvalue(graph,A, tol):
    # Initialize random vector
    x = np.random.rand(len(graph))
   
    # Calculate adjacency matrix and normalize to obtain transition matrix
    #D = np.diag(np.sum(A, axis=1))
   # P = np.linalg.inv(D).dot(A)

    lambda_1_old = 0
    #old_x = x
    while True:
        x = power_iteration(A, x)
        a = np.dot(x,A)
        b = np.dot(a, x)
        lambda_1 = b / np.dot(x,x)

        if abs(lambda_1 - lambda_1_old) < tol:
       # if abs(np.linalg.norm(x-old_x)) < tol:
            break
        lambda_1_old = lambda_1
        #old_x = x
        
    return lambda_1, x

def projection(u, v):
    # calculate dot product of u and v
        dot_uv = np.dot(u, v)

    # calculate dot product of u with itself
        dot_uu = np.dot(u, u)

    # calculate projection of v onto u
        proj_uv = (dot_uv / dot_uu) * u
        return proj_uv

def get_2nd_highest_eigenvalue(graph,A,u, tol):
    w = np.random.rand(len(graph))
    proj_uv = projection(u, w)
    u0 = w - proj_uv

    #D = np.diag(np.sum(A, axis=1))
    #P = np.linalg.inv(D).dot(A)

    #lambda_2_old = 0
    old_u = u0
    while True:#CRASHES FOR CLIQUE GRAPH because wt = Pu(wt) so the vt = 0
        #to find new u
        wt = np.dot(A, old_u)
        vt = wt - projection(u, wt)
        
        #print(np.linalg.norm(vt))
        new_u = vt / np.linalg.norm(vt)#קבוע הכי קטן

        #to get the eigenvalue
        a = np.dot(new_u,A)
        b = np.dot(a, new_u)
        lambda_2 = b / np.dot(new_u,new_u)
        #if abs(lambda_2 - lambda_2_old) < tol:
        if abs(np.linalg.norm(new_u - old_u)) < tol:
            break
        #lambda_2_old = lambda_2
        old_u = new_u
        
    return lambda_2


def get_2_highest(G, A, tol):
    lambda_1, x = get_highest_eigenvalue(G, A, tol)
    lambda_2 = get_2nd_highest_eigenvalue(G, A, x, tol)
    return lambda_1, lambda_2
'''


'''

print("for two cliques:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(two_cliques_G, norm_mat_two_cliques, tol)
    print("from power function:", first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)

print("for candy:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(candyG, norm_mat_candy, tol)
    print("from power function:", first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)


print("for clique:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(cliqueG, norm_mat_clique, tol)
    print("from power function:", first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)

print("for ring:")
for k in range(2, 7):
    tol = 2 ** -k
    first, second = get_2_highest(ringG, norm_mat_ring, tol)
    print("from power function:",first, second)
    print("Tolerance level:", tol, "Ratio of first and second eigenvalues:", first/second)




w, v = np.linalg.eig(norm_mat_ring)
idx = np.argsort(w)[::-1]  # Get indices that sort eigenvalues in descending order
w = w[idx]
v = v[:, idx]
print("from eig function:", w[0], w[1])
