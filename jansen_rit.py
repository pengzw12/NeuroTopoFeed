"""
jansen_rit.py
Coupled Jansen-Rit Simulation Platform for Cognitive State Generation

Implements the Jansen-Rit neural mass model with phenomenological parameter
adjustments to simulate Focus, Relax, Anxiety, and Fatigue states.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.integrate import odeint


class JansenRitNeuralMass:
    """
    Single Jansen-Rit neural mass model (Jansen & Rit, 1995).
    
    Three populations: pyramidal (y0, y1), excitatory interneuron (y2),
    inhibitory interneuron (y3).
    
    Parameters
    ----------
    A : float
        Excitatory gain (mV).
    B : float
        Inhibitory gain (mV).
    a : float
        Excitatory inverse time constant (1/ms).
    b : float
        Inhibitory inverse time constant (1/ms).
    v0 : float
        Firing threshold (mV).
    e0 : float
        Maximum firing rate (1/ms).
    r : float
        Firing gain slope (1/mV).
    """
    
    def __init__(self, A=3.25, B=22.0, a=100.0, b=50.0, v0=6.0, e0=2.5, r=0.56):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.v0 = v0
        self.e0 = e0
        self.r = r
    
    def _sigmoid(self, v):
        """Firing rate sigmoid."""
        return 2 * self.e0 / (1 + np.exp(self.r * (self.v0 - v)))
    
    def _rhs(self, y, t, input_pA=0):
        """
        ODE right-hand side.
        y = [y0, y1, y2, y3, y4, y5] where:
        y0, y1: pyramidal cell (membrane potential + derivative)
        y2, y3: excitatory interneuron
        y4, y5: inhibitory interneuron
        """
        y0, y1, y2, y3, y4, y5 = y
        
        # Input noise (phenomenological)
        noise = np.random.normal(0, input_pA / 1000.0)  # scale appropriately
        
        # Sigmoid inputs
        S_pyramid = self._sigmoid(y1 - y3)  # pyramidal receives exc - inh
        S_exc = self._sigmoid(y0)
        S_inh = self._sigmoid(y0)
        
        dy0 = y1
        dy1 = self.a * (self.A * S_exc + self.A * noise) - 2 * self.a * y1 - self.a**2 * y0
        dy2 = y3
        dy3 = self.a * (self.A * S_pyramid) - 2 * self.a * y3 - self.a**2 * y2
        dy4 = y5
        dy5 = self.b * (self.B * S_inh) - 2 * self.b * y5 - self.b**2 * y4
        
        return [dy0, dy1, dy2, dy3, dy4, dy5]
    
    def simulate(self, duration=5.0, dt=0.001, input_pA=30.0):
        """
        Simulate single neural mass.
        
        Returns
        -------
        t : ndarray
            Time vector.
        signal : ndarray
            Simulated EEG-like signal (pyramidal output y1-y3).
        """
        t = np.arange(0, duration, dt)
        y0_init = np.random.randn(6) * 0.1
        
        sol = odeint(self._rhs, y0_init, t, args=(input_pA,))
        signal = sol[:, 1] - sol[:, 3]  # pyramidal output
        return t, signal


class CoupledJansenRitNetwork:
    """
    Coupled Jansen-Rit network for cognitive state simulation.
    
    N nodes with exponential distance-based connectivity (Table 5).
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes (default 64 for 10-20 system).
    coords : ndarray, shape (n_nodes, 3)
        Electrode coordinates (projected to 2D scalp map).
    lambda_decay : float
        Spatial decay constant (mm).
    """
    
    STATE_PRESETS = {
        'Focus':   {'A_scale': 1.20, 'B_scale': 1.00, 'sigma_scale': 0.70},
        'Relax':   {'A_scale': 1.00, 'B_scale': 1.10, 'sigma_scale': 1.00},
        'Anxiety': {'A_scale': 1.40, 'B_scale': 0.80, 'sigma_scale': 1.50},
        'Fatigue': {'A_scale': 0.70, 'B_scale': 1.30, 'sigma_scale': 0.80},
    }
    
    def __init__(self, n_nodes=64, coords=None, lambda_decay=20.0):
        self.n_nodes = n_nodes
        self.lambda_decay = lambda_decay
        
        # Generate coordinates if not provided (simplified 10-20 layout)
        if coords is None:
            self.coords = self._generate_1020_coords(n_nodes)
        else:
            self.coords = coords
        
        # Build connection matrix (Equation in Section 3.4)
        self.W = self._build_connectivity()
        
        # Initialize nodes with physiological parameter distributions
        self.nodes = []
        for _ in range(n_nodes):
            A = np.random.normal(3.25, 0.5)
            B = np.random.normal(22.0, 2.0)
            node = JansenRitNeuralMass(A=A, B=B)
            self.nodes.append(node)
    
    def _generate_1020_coords(self, n_nodes):
        """Generate simplified 10-20 system coordinates on scalp."""
        # Simplified: evenly distributed on unit circle for demo
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        r = 0.5 + 0.5 * np.random.rand(n_nodes)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack([x, y, np.zeros(n_nodes)])
    
    def _build_connectivity(self):
        """Build distance-based connection matrix (Section 3.4)."""
        W = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                d = np.linalg.norm(self.coords[i, :2] - self.coords[j, :2]) * 100  # scale to mm
                if d < 40:
                    W[i, j] = np.exp(-d / self.lambda_decay)
        return W
    
    def simulate_state(self, state_name, duration=5.0, dt=0.004):
        """
        Simulate specified cognitive state (Table 5).
        
        Parameters
        ----------
        state_name : str
            'Focus', 'Relax', 'Anxiety', or 'Fatigue'.
        duration : float
            Simulation duration (seconds).
        dt : float
            Time step (seconds). 250 Hz = 0.004s.
            
        Returns
        -------
        signals : ndarray, shape (n_nodes, n_samples)
            Multi-channel simulated EEG.
        labels : ndarray
            State label for each sample.
        """
        preset = self.STATE_PRESETS[state_name]
        n_samples = int(duration / dt)
        
        signals = np.zeros((self.n_nodes, n_samples))
        
        # Initialize node states
        states = [np.random.randn(6) * 0.1 for _ in range(self.n_nodes)]
        
        t = np.arange(0, duration, dt)
        for s_idx in range(n_samples):
            for i in range(self.n_nodes):
                # Adjust input noise based on state
                sigma = 30.0 * preset['sigma_scale']
                
                # Update node with coupling from neighbors
                coupling = 0
                for j in range(self.n_nodes):
                    if i != j and self.W[i, j] > 0:
                        coupling += self.W[i, j] * states[j][1]  # coupled via pyramidal output
                
                # One step integration (simplified Euler)
                dy = self.nodes[i]._rhs(states[i], t[s_idx], input_pA=sigma)
                dy = np.array(dy) + np.array([0, coupling * 0.001, 0, 0, 0, 0])
                states[i] += np.array(dy) * dt
                signals[i, s_idx] = states[i][1] - states[i][3]
        
        labels = np.full(n_samples, state_name)
        return signals, labels
    
    def generate_dataset(self, trials_per_state=25, duration=5.0):
        """
        Generate full 4-class dataset for algorithm validation.
        
        Returns
        -------
        X : list of ndarray
            Trial signals.
        y : list of str
            State labels.
        """
        X, y = [], []
        for state in self.STATE_PRESETS.keys():
            for _ in range(trials_per_state):
                sig, lab = self.simulate_state(state, duration)
                X.append(sig)
                y.append(state)
        return X, y


# === Example usage ===
if __name__ == '__main__':
    np.random.seed(42)
    network = CoupledJansenRitNetwork(n_nodes=64)
    
    # Generate one trial for each state
    for state in ['Focus', 'Relax', 'Anxiety', 'Fatigue']:
        sig, lab = network.simulate_state(state, duration=2.0)
        print(f"{state}: shape={sig.shape}, mean={np.mean(sig):.3f}, std={np.std(sig):.3f}")
    
    # Full dataset
    X, y = network.generate_dataset(trials_per_state=10, duration=2.0)
    print(f"\nDataset: {len(X)} trials")
    from collections import Counter
    print(f"Label distribution: {Counter(y)}")
