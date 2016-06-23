import matplotlib.pyplot as plt
import networkx as nx
from .modularity import mod_labels


def compare_clusterings(G,Q,c1,c2):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.set_size_inches(16, 8)

    pos = nx.spring_layout(G)
    for ax,c in [(ax1,c1),(ax2,c2)]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        nx.draw_networkx(G, ax=ax,pos=pos, with_labels=False, node_color=c, cmap=plt.cm.Blues)
        ax.set_title("mod = %f" % mod_labels(Q,c))
