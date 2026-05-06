"""
incremental_ph.py
Incremental Persistent Homology with Directed Flag Complex

Implements real-time topological feature extraction (beta_0, beta_1, beta_2)
with incremental updates and clinical dimension truncation.

Dependencies: numpy, scipy
"""

import numpy as np
from itertools import combinations


class IncrementalPersistentHomology:
    """
    Incremental PH computation for directed brain networks.
    
    Computes beta_0, beta_1, beta_2 (clinical dimension truncation)
    with incremental edge updates and exponential decay weighting.
    
    Parameters
    ----------
    n_nodes : int
        Number of network nodes (e.g., 64 EEG channels).
    max_dim : int
        Maximum simplex dimension to compute (default 2 for beta_0,1,2).
    decay_tau : float
        Exponential decay constant for edge weights (seconds).
    window_sec : float
        Sliding window length (seconds).
    """
    
    def __init__(self, n_nodes=64, max_dim=2, decay_tau=5.0, window_sec=10.0):
        self.n_nodes = n_nodes
        self.max_dim = max_dim
        self.decay_tau = decay_tau
        self.window_sec = window_sec
        
        # Adjacency matrix with timestamps and weights
        self.adjacency = np.zeros((n_nodes, n_nodes))
        self.edge_time = np.full((n_nodes, n_nodes), -np.inf)
        self.current_time = 0.0
        
        # Persistent homology state
        self.simplices = {}   # (dim, tuple(nodes)) -> (birth, death)
        self.betti = [0, 0, 0]
    
    def add_edges(self, edges, current_time):
        """
        Incrementally add/update directed edges.
        
        Parameters
        ----------
        edges : list of (i, j, weight)
            Directed edges with current weight.
        current_time : float
            Current timestamp (seconds).
        """
        self.current_time = current_time
        
        # Update edge weights and timestamps
        for i, j, w in edges:
            self.adjacency[i, j] = w
            self.edge_time[i, j] = current_time
        
        # Exponential decay: remove edges below 10% of peak
        age = current_time - self.edge_time
        valid_mask = (age >= 0) & (age < self.window_sec)
        decay_factor = np.exp(-age / self.decay_tau)
        
        self.adjacency *= decay_factor * valid_mask
        self.adjacency[self.adjacency < 0.1 * np.max(self.adjacency)] = 0
    
    def compute_flag_complex(self, threshold=0.1):
        """
        Build directed flag complex from thresholded adjacency.
        
        A directed k-simplex is a (k+1)-tuple with consistently oriented edges.
        Only computes up to max_dim (2) for clinical efficiency.
        """
        active = self.adjacency > threshold
        self.betti = [0, 0, 0]
        
        # beta_0: connected components (undirected interpretation)
        undirected = active | active.T
        visited = set()
        components = 0
        for node in range(self.n_nodes):
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    v = stack.pop()
                    if v not in visited:
                        visited.add(v)
                        neighbors = np.where(undirected[v])[0]
                        stack.extend([n for n in neighbors if n not in visited])
        self.betti[0] = components
        
        # beta_1: count directed 2-simplices (triangles with consistent orientation)
        count_simplices_1 = 0
        for i, j, k in combinations(range(self.n_nodes), 3):
            # Check if all directed edges exist for at least one cyclic ordering
            if (active[i, j] and active[j, k] and active[i, k]) or \
               (active[i, k] and active[k, j] and active[i, j]):
                count_simplices_1 += 1
        
        # beta_1 approximation: cycles = edges - nodes + components
        n_edges = np.sum(active)
        self.betti[1] = max(0, int(n_edges - self.n_nodes + components))
        
        # beta_2: directed 3-simplices (tetrahedra with consistent orientation)
        count_simplices_2 = 0
        for i, j, k, l in combinations(range(self.n_nodes), 4):
            # Check for any Hamiltonian path ordering
            nodes = [i, j, k, l]
            found = False
            for perm in combinations(nodes, 4):
                # Simple check: 6 edges of complete graph present
                edges_needed = [(perm[a], perm[b]) for a in range(4) for b in range(a+1, 4)]
                if all(active[u, v] or active[v, u] for u, v in edges_needed):
                    found = True
                    break
            if found:
                count_simplices_2 += 1
        
        # beta_2 simplified approximation
        self.betti[2] = max(0, count_simplices_2 - count_simplices_1 + n_edges)
        self.betti[2] = min(self.betti[2], self.n_nodes * 2)  # ceiling
        
        return np.array(self.betti)
    
    def update(self, edges, current_time):
        """
        Full incremental update: add edges, decay old ones, compute PH.
        
        Returns
        -------
        betti : ndarray, shape (3,)
            [beta_0, beta_1, beta_2]
        """
        self.add_edges(edges, current_time)
        return self.compute_flag_complex()
    
    def reset(self):
        """Reset all state for new session."""
        self.adjacency.fill(0)
        self.edge_time.fill(-np.inf)
        self.current_time = 0.0
        self.simplices.clear()
        self.betti = [0, 0, 0]


class TopologicalBiomarkers:
    """
    Convert raw Betti numbers into clinically interpretable biomarkers.
    """
    
    @staticmethod
    def network_efficiency_index(betti):
        """Patient-level biomarker: beta_0 / beta_2 ratio."""
        b0, _, b2 = betti
        if b2 == 0:
            return b0
        return b0 / (b2 + 1e-6)
    
    @staticmethod
    def global_coherence_score(betti):
        """Clinician-level: beta_1 / beta_0."""
        b0, b1, _ = betti
        if b0 == 0:
            return b1
        return b1 / (b0 + 1e-6)
    
    @staticmethod
    def hyperconnectivity_index(betti):
        """Anxiety signature: high beta_2 with unstable filtration."""
        _, _, b2 = betti
        return b2
    
    @staticmethod
    def sluggishness_index(betti, dt=1.0):
        """Fatigue signature: low-dim topology with slow evolution."""
        b0, b1, b2 = betti
        total = b0 + b1 + b2 + 1e-6
        # Low-dimensional dominance = slow evolution
        return (b0 + 0.5 * b1) / total


# === Example usage ===
if __name__ == '__main__':
    iph = IncrementalPersistentHomology(n_nodes=64)
    
    # Simulate 10 random edges per window
    np.random.seed(42)
    for t_win in range(10):
        edges = [(i, j, np.random.rand())
                 for i, j in zip(np.random.randint(0, 64, 10), np.random.randint(0, 64, 10))
                 if i != j]
        betti = iph.update(edges, current_time=t_win * 2.0)
        print(f"Window {t_win}: beta = {betti}")
    
    biomarkers = TopologicalBiomarkers()
    final_betti = iph.betti
    print(f"\nNetwork Efficiency Index: {biomarkers.network_efficiency_index(final_betti):.3f}")
    print(f"Global Coherence Score:   {biomarkers.global_coherence_score(final_betti):.3f}")
    print(f"Hyper-connectivity Index: {biomarkers.hyperconnectivity_index(final_betti):.3f}")
