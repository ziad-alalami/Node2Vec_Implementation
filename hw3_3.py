import numpy as np
import sys

def sorted_insert(lst, element):

    for i in range(len(lst)):
        if element < lst[i]:
            lst.insert(i, element)
            return lst
    lst.append(element)
    return lst

# Graph class
class Graph:

    # Store the graph as set of adjacency lists of each nodes
    # Assume undirected graph
    # For example, if we have node 1, node 2 and edge (1, 2), (1, 3),
    # adj_list = {1: [2, 3], 2: [1], 3: [1]} 
    def __init__(self):
        self.adj_list = {}

    # Add an edge to the graph (i.e., update adj_list)
    def add_edge(self, u, v):
        self.adj_list[u] = sorted_insert(self.adj_list.get(u, []), v)
        self.adj_list[v] = sorted_insert(self.adj_list.get(v, []), u)


    # Return the neighbors of a node
    def neighbors(self, node):
        return self.adj_list[node]



# This random_walk funtion is a hint 
# Refer to this function to implement the node2vec_walk function below
def random_walk(graph, start, length=5):
    # Random Walk
    walk = [start]
    for _ in range(length - 1):
        neighbors = graph.neighbors(walk[-1])
        if not neighbors:
            break
        walk.append(np.random.choice(neighbors))
    return walk



# Implement node2vec algorithm in DFS
# The length of the walk is fixed to 5
# When sampling next node, visit the node with the smallest index
# Note that it returns the trajectory of walker
# so that same node can be visited multiple times
def node2vec_walk_dfs(graph, start, length=5):
    walk = [start]
    stack = [start]

    while len(walk) < length:
        if not stack:
            break

        current = stack[-1]
        neighbors = sorted(graph.neighbors(current))

        found = False
        for neighbor in neighbors:
            if neighbor not in walk:
                stack.append(neighbor)
                walk.append(neighbor)
                found = True
                break

        if not found:
            stack.pop()
            if stack:
                walk.append(stack[-1])

    return walk



# Train W1, W2 matrices using Skip-Gram
# The window size of fixed to 2, which means you should check each 2 nodes before and after the center node.  
# - Ex. Assume we have walk sequence [1, 2, 3, 4, 5].
# - If center node is 3, we should consider [1, 2, 3, 4, 5] as the context nodes.
# - If center node is 2, we should consider [1, 2, 3, 4] as the context nodes.
# Repeat the training process for 3 epochs, with learning rate 0.01
# Use softmax function when computing the loss
def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)
    for e in range(epochs):
        for walk in walks:
            for i, v in enumerate(walk):
                start = max(0, i - window)
                end = min(len(walk), i + window + 1)
                context_nodes = walk[start:i] + walk[i + 1:end]
                for w in context_nodes:
                    w_one = np.zeros((n_nodes,1))
                    w_one[w] = 1

                    x_one = np.zeros((n_nodes,1))
                    x_one[v] = 1

                    dW1, dW2 = compute_gradients(W1, W2,x_one, w_one)
                    #dW2 /= len(context_nodes)
                    #dW1 /= len(context_nodes)

                    W1 -= lr * dW1
                    W2 -= lr * dW2

    Z_5 = np.eye(n_nodes)[4] @ W1
    Z_10 = np.eye(n_nodes)[9] @ W1
    return Z_5, Z_10


# You can freely define your functions/classes if you want
def softmax(output):
    #max_exp = np.max(output, axis = 1, keepdims= True)
    stable_exp = np.exp(output)
    sum_col = np.sum(stable_exp)
    top = stable_exp / sum_col
    #index_ = np.argmax(top)
    #return index_
    return top
def loss(output):
    return -np.log(softmax(output))


def compute_gradients(W1, W2, x, w):
    z = W1.T @ x
    s = W2.T @ z
    s.reshape((1,-1))
    y = softmax(s)
    grad_s = y - w

    #y_diag = np.diagflat(y)
    #softmax_jacobian = y_diag - y @ y.T

    #grad_s = softmax_jacobian @ grad_y

    dL_dW2 = np.outer(z, grad_s)
    dL_dW1 = np.outer(x, W2 @ grad_s)
    #dL_dW2 = z @ grad_s.T

    #grad_z = W2 @ grad_s

    #dL_dW1 = x @ grad_z.T

    #dL_dW2 = np.outer(z, grad_logits)
    #dL_dW1 = np.outer(x, W2 @ grad_logits)
    return dL_dW1, dL_dW2

# Main function
def main():

    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)


    # Create graph
    graph = Graph()


    # Edges list
    # Note that the edges are undirected, and node idx starts with 1
    # ex. edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]  
    edges = []


    # Parse edges from the command line file path
    with open(sys.argv[1], "r") as file:
        content = file.readlines()
        for line in content:
            line_ = line.split()
            edges.append((int(line_[0]) - 1, int(line_[1]) - 1))




    # Update graph
    for edge in edges:
        graph.add_edge(*edge)


    # Generate random walks on DFS
    walks_dfs = [node2vec_walk_dfs(graph, node) for node in graph.adj_list]


    # Train Skip-Gram on DFS
    embeddings_dfs = train_skipgram(walks_dfs, len(graph.adj_list))


    # Print the embeddings ===========================================
    print(f"{embeddings_dfs[0][0]:.5f}")
    print(f"{embeddings_dfs[1][0]:.5f}")


if __name__ == "__main__":
    main()
