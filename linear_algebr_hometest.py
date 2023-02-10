import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

cliqueG = clique(7)

fig, ax = plt.subplots()
nx.draw(cliqueG, with_labels=True, ax=ax)
ax.set_title("Clique Graph")

plt.savefig("clique_graph.png")

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

fig, ax = plt.subplots()
nx.draw(ringG, with_labels=True, ax=ax)
ax.set_title("Ring Graph")

plt.savefig("ring_graph.png")


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

fig, ax = plt.subplots()
nx.draw(candyG, with_labels=True, ax=ax)
ax.set_title("Candy Graph")
plt.savefig("candy_graph.png")

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
fig, ax = plt.subplots()
nx.draw(two_cliques_G, with_labels=True, ax=ax)
ax.set_title("Two Cliques Graph")
plt.savefig("two_cliques_graph.png")
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
    # Create the adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    
    # Normalize the rows to obtain the transition matrix
    transition_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis=1)
    
    return transition_matrix

def stationary_distribution(transition_matrix):
    # Find the eigenvector corresponding to the eigenvalue of 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    eigenvector = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real
   
    # Normalize the eigenvector
    stationary_dist = eigenvector / eigenvector.sum()
    
    return stationary_dist
adjacency_mat_candy = nx.to_numpy_array(candyG)
tran_mat_candy = transition_matrix(candyG)
stationary_candy = stationary_distribution(tran_mat_candy)

adjacency_mat_clique = nx.to_numpy_array(cliqueG)
tran_mat_clique = transition_matrix(cliqueG)
stationary_clique = stationary_distribution(tran_mat_clique)

adjacency_mat_ring = nx.to_numpy_array(ringG)
tran_mat_ring = transition_matrix(ringG)
stationary_ring = stationary_distribution(tran_mat_ring)

adjacency_mat_two_cliques = nx.to_numpy_array(two_cliques_G)
tran_mat_two_cliques = transition_matrix(two_cliques_G)
stationary_two_cliques = stationary_distribution(tran_mat_two_cliques)

'''
print("stationary candy: ")
for i in stationary_candy:
    print(i)

print("stationary ring: ")
for i in stationary_ring:
    print(i)

print("stationary two cliques: ")
for i in stationary_two_cliques:
    print(i)

print("stationary clique: ")
for i in stationary_clique:
    print(i)
'''

#question 3
def generalized_power_method(A, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    for i in range(max_iterations):
        y = A @ x
        eigenvalue = np.dot(x, y)
        if np.linalg.norm(y) == 0:
            break
        x = y / np.linalg.norm(y)
        if np.abs(eigenvalue - np.dot(x, A @ x)) < tolerance:
            break
    return eigenvalue

def ratio_of_first_two_eigenvalues(A):
    eigenvalue1 = generalized_power_method(A)
    A = A - eigenvalue1 * np.outer(np.ones(A.shape[0]), np.ones(A.shape[0]))
    eigenvalue2 = generalized_power_method(A)
    return eigenvalue1 / eigenvalue2

print("two cliques: " + str(ratio_of_first_two_eigenvalues(adjacency_mat_two_cliques)))
print("candy: " + str(ratio_of_first_two_eigenvalues(adjacency_mat_candy)))
print("ring: " + str(ratio_of_first_two_eigenvalues(adjacency_mat_ring)))
print("clique: " + str(ratio_of_first_two_eigenvalues(adjacency_mat_clique)))
#יחס שלילי, יוצא שהערך העצמי השני שיוצא בכל מצב הוא שלילי בעוד שהראשון חיובי ונראה תקין
