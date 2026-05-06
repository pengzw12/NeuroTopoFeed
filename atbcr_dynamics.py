"""
atbcr_dynamics.py
ATBCR-Inspired Neural Dynamics with Error-Monitored Hybrid Predictor

Implements state-dependent prediction through ATBCR linearization
with automatic fallback to full Kuramoto integration when error exceeds threshold.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.integrate import odeint


class ATBCRKuramotoDynamics:
    """
    Hybrid predictor: ATBCR fast linearization + full Kuramoto fallback.
    
    Uses extended Kuramoto model (Equation 4) with state-dependent coupling.
    Switching threshold: 0.05 (normalized phase error).
    
    Parameters
    ----------
    n_nodes : int
        Number of neural populations (e.g., 64).
    omega : ndarray
        Intrinsic frequencies (rad/s).
    mu : float
        Coupling strength.
    """
    
    def __init__(self, n_nodes=64, omega=None, mu=0.03):
        self.n_nodes = n_nodes
        self.omega = omega if omega is not None else np.random.normal(10, 2, n_nodes)
        self.mu = mu
        self.epsilon = 0.3      # Confidence threshold
        self.theta_pol = 0.5    # Polarization threshold
        
        # Error monitoring buffer
        self.error_buffer = []
        self.buffer_size = 5
        self.current_model = 'atbcr'
        self.theta = np.random.rand(n_nodes) * 2 * np.pi
    
    def _coupling(self, dtheta):
        """State-dependent coupling function (Equation 4)."""
        abs_dtheta = np.abs(dtheta)
        if abs_dtheta < self.epsilon:
            return self.mu      # attraction
        elif abs_dtheta > self.theta_pol:
            return -self.mu     # repulsion
        else:
            return 0.0          # neutral zone
    
    def _kuramoto_rhs(self, theta, t):
        """Full Kuramoto ODE right-hand side."""
        dtheta = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            coupling_sum = 0.0
            for j in range(self.n_nodes):
                if i == j:
                    continue
                delta = theta[j] - theta[i]
                K_ij = self._coupling(delta)
                coupling_sum += K_ij * np.sin(delta)
            dtheta[i] = self.omega[i] + coupling_sum
        return dtheta
    
    def _atbcr_linearized_step(self, theta_prev, dt=0.001):
        """
        ATBCR linearized prediction (Equation 4, first-order approximation).
        
        Taylor expansion around current state: dtheta/dt ≈ f(theta) + J_f(theta)(theta_new - theta)
        For small dt: theta(t+dt) ≈ theta(t) + dt * f(theta(t))
        """
        f = self._kuramoto_rhs(theta_prev, 0)
        theta_new = theta_prev + dt * f
        return theta_new % (2 * np.pi)
    
    def _full_kuramoto_step(self, theta_prev, dt=0.001):
        """
        Full Kuramoto integration using RK4 (more accurate than Euler).
        """
        def rhs(y, t):
            return self._kuramoto_rhs(y, t)
        
        t_span = [0, dt]
        sol = odeint(rhs, theta_prev, t_span)
        return sol[-1] % (2 * np.pi)
    
    def predict(self, dt=0.001):
        """
        Predict next state using hybrid model.
        
        Returns
        -------
        theta_new : ndarray
            Predicted phase vector.
        model_used : str
            'atbcr' or 'kuramoto'.
        """
        # Always compute both for error monitoring
        theta_atbcr = self._atbcr_linearized_step(self.theta, dt)
        theta_kuramoto = self._full_kuramoto_step(self.theta, dt)
        
        # Normalized phase error (Theorem 1 metric)
        error = np.mean(np.abs(theta_kuramoto - theta_atbcr)) / (2 * np.pi)
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.buffer_size:
            self.error_buffer.pop(0)
        
        # Switching logic (from Appendix A.2)
        avg_error = np.mean(self.error_buffer[-3:]) if len(self.error_buffer) >= 3 else error
        
        if avg_error > 0.05:
            self.current_model = 'kuramoto'
            self.theta = theta_kuramoto
            return theta_kuramoto, 'kuramoto'
        elif np.mean(self.error_buffer[-5:]) < 0.05 and len(self.error_buffer) >= 5:
            self.current_model = 'atbcr'
            self.theta = theta_atbcr
            return theta_atbcr, 'atbcr'
        else:
            # Default to current model
            if self.current_model == 'kuramoto':
                self.theta = theta_kuramoto
                return theta_kuramoto, 'kuramoto'
            else:
                self.theta = theta_atbcr
                return theta_atbcr, 'atbcr'
    
    def online_parameter_update(self, Q_t, H_spectral, Q_min=0.1, Q_max=0.8, H_max=2.5):
        """
        Update epsilon and theta_pol every 30 seconds (Equation 10, 11).
        
        Parameters
        ----------
        Q_t : float
            Current modularity (Louvain).
        H_spectral : float
            Current spectral entropy.
        """
        # Equation (10) with clip to [0.1, 0.6]
        self.epsilon = np.clip(
            0.3 - 0.5 * (Q_t - Q_min) / (Q_max - Q_min),
            0.1, 0.6
        )
        # Equation (11) with clip to [0.2, 0.8]
        self.theta_pol = np.clip(
            0.4 + 0.8 * H_spectral / H_max,
            0.2, 0.8
        )
    
    def get_state_signature(self):
        """Map current phase to cognitive state features."""
        # Order parameter: synchronization degree
        r = np.abs(np.mean(np.exp(1j * self.theta)))
        return {
            'sync_index': r,
            'mean_phase': np.mean(self.theta),
            'phase_variance': np.var(self.theta),
            'current_model': self.current_model,
            'last_error': self.error_buffer[-1] if self.error_buffer else 0
        }


def compute_error_bound(mu, N, t, max_delta_theta, min_omega):
    """
    Theorem 1: ATBCR Linearization Error Bound (Equation 5).
    
    Returns normalized phase error upper bound.
    """
    # time normalized by characteristic scale 1/min_omega
    t_norm = t * min_omega
    bound = (mu**2 * N**2 * t_norm / (12 * np.pi)) * max(abs(np.cos(max_delta_theta) - 1), 0)
    return bound


# === Example usage ===
if __name__ == '__main__':
    np.random.seed(42)
    dyn = ATBCRKuramotoDynamics(n_nodes=64, mu=0.03)
    
    # Simulate 1 second
    dt = 0.001
    n_steps = 1000
    
    model_usage = {'atbcr': 0, 'kuramoto': 0}
    for _ in range(n_steps):
        _, model = dyn.predict(dt)
        model_usage[model] += 1
    
    print(f"Model usage: ATBCR={model_usage['atbcr']} ({model_usage['atbcr']/n_steps*100:.1f}%), "
          f"Kuramoto={model_usage['kuramoto']} ({model_usage['kuramoto']/n_steps*100:.1f}%)")
    
    # Theorem 1 error bound check
    bound = compute_error_bound(mu=0.03, N=64, t=0.5, max_delta_theta=0.3, min_omega=8.0)
    print(f"Theorem 1 error bound (mu=0.03, t=0.5s): {bound:.6f}")
    print(f"Safe threshold: 0.05 -> {'PASS' if bound < 0.05 else 'SWITCH to Kuramoto'}")
