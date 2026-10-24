from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]

import pickle
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# 1. Read the directed graph from pickle.
# -------------------------
with open(str(_PROJECT_ROOT / 'NFV/p_net_directed2.pkl'), "rb") as f:
    G = pickle.load(f)

# -------------------------
# 2. Basic checks.
# -------------------------
print("Graph type:", type(G))
print("Is directed:", G.is_directed())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# -------------------------
# 3. Choose a layout, preferably spring or kamada_kawai.
# -------------------------
pos = nx.spring_layout(G, seed=42)   # Reproducible layout.
# pos = nx.kamada_kawai_layout(G)    # Alternative, more uniform layout.

# -------------------------
# 4. Draw the directed graph.
# -------------------------
plt.figure(figsize=(10, 8))

nx.draw_networkx_nodes(
    G, pos,
    node_size=500,
    node_color="lightblue"
)

nx.draw_networkx_labels(
    G, pos,
    font_size=10
)

nx.draw_networkx_edges(
    G, pos,
    arrowstyle="->",
    arrowsize=15,
    width=1.2,
    edge_color="gray"
)

plt.title("Directed Network Graph")
plt.axis("off")
plt.tight_layout()
plt.show()