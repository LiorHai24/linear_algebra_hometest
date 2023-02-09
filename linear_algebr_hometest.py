import networkx as nx
import matplotlib.pyplot as plt


n = 5
clique = nx.complete_graph(n)


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
