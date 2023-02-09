import networkx as nx
import matplotlib.pyplot as plt

n = 5
clique = nx.complete_graph(n)


def ring(n):
    G = nx.Graph()
    nodes = list(range(n))
    edges = [(i, i) for i in range(n)]+ [(i, (i + 1) % n) for i in range(n)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

ringG = ring(7)

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

nx.draw(candyG, with_labels=True)
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
            if i != j:
                G.add_edge(i, j)


    # Connect the clique vertices
    for i in range((3*n//4) +1, n+1):
        for j in range((3*n//4) +1, n+1):
            if i != j:
                G.add_edge(i, j)


    G.add_edge(1, (n//2)+1)
    G.add_edge(n//2, (3*n//4) +1)
    return G

two_cliques_G = two_cliques(8)
nx.draw(two_cliques_G, with_labels=True)
plt.savefig("two_cliques_graph.png")
