import numpy as np
import scipy as sp
import networkx as nx
import modsdp as msdp

networks = {"karate" : "data/karate/karate.gml",
            "polbooks" : "data/polbooks/polbooks.gml",
            "football" : "data/football/football.gml",
            "dolphins" : "data/dolphins/dolphins.gml",
            "neural" : "data/celegansneural/celegansneural.gml",
            "netscience" : "data/netscience/netscience.gml"
}




def sdp_experiment(l=5000,solver='rbr'):
    k = 10
    print("sdp using solver %s, k = %d " % (solver,k))
    for f in networks:
        G = nx.read_gml(networks[f])
        W = nx.adjacency_matrix(G).astype(float)
        Q = msdp.modmatrix(W)

        print("%s %d " % (f, len(G.nodes())), end="")
        for r in ['gw2','gw','by','frieze']:
            c = msdp.sdp(W,k,solver=solver,rounding=r,round_iters=l)
            print("(%s %f) " % (r, msdp.mod_labels(Q,c)), end="")
        c = msdp.spectral(W,k,round_iters=100)
        print("(spec %f)" % msdp.mod_labels(Q,c))

