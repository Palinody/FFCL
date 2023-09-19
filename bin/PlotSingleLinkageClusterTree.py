import matplotlib.pyplot as plt
import networkx as nx

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Create a binary tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

# Initialize a directed graph
G = nx.DiGraph()

def build_graph(node, parent=None, pos=None, level=0):
    if node:
        if not pos:
            pos = {node: (0, 0)}
        else:
            x, y = pos[parent]
            if parent.left == node:
                x -= 1 / (level + 1)
            else:
                x += 1 / (level + 1)
            y -= 1
            pos[node] = (x, y)

        if parent:
            G.add_edge(parent, node)
        
        build_graph(node.left, node, pos, level + 1)
        build_graph(node.right, node, pos, level + 1)

# Build the graph
build_graph(root)

# Draw the tree
pos = nx.spring_layout(G, seed=42)  # You can use different layout algorithms
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', arrows=False)
plt.axis('off')
plt.show()
