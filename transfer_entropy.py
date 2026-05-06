"""
transfer_entropy.py
Adaptive Transfer Entropy Estimation with Spectral-Entropy-Based History Length

Implements transfer entropy with adaptive history length selection based on
spectral entropy (Equation 7-8).

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.signal import welch


def compute_spectral_entropy(signal, fs=250.0, nperseg=256):
    """
    Compute spectral entropy of a signal (Equation 7).
    
    H_spectral = -sum_f P(f) * log P(f)
    
    Parameters
    ----------
    signal : ndarray
        1D time series.
    fs : float
        Sampling frequency.
    nperseg : int
        FFT segment length.
        
    Returns
    -------
    H : float
        Spectral entropy.
    H_max : float
        Maximum possible entropy (log n_freqs).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    # Normalize to probability distribution
    psd = psd + 1e-12  # avoid log(0)
    P = psd / np.sum(psd)
    H = -np.sum(P * np.log(P + 1e-12))
    H_max = np.log(len(P))
    return H, H_max


def adaptive_history_length(signal, k_min=2, k_max=8, fs=250.0):
    """
    Compute adaptive history length k based on spectral entropy (Equation 8).
    
    k = k_min + floor((k_max - k_min) * (1 - H_spectral / H_max))
    
    High entropy (disordered) -> short memory (k_min)
    Low entropy (ordered) -> long memory (k_max)
    
    Parameters
    ----------
    signal : ndarray
        1D time series for entropy estimation.
    k_min, k_max : int
        History length bounds.
    fs : float
        Sampling frequency.
        
    Returns
    -------
    k : int
        Adaptive history length.
    H_spectral : float
        Spectral entropy value.
    """
    H, H_max = compute_spectral_entropy(signal, fs=fs)
    k = k_min + int(np.floor((k_max - k_min) * (1.0 - H / H_max)))
    k = np.clip(k, k_min, k_max)
    return k, H


def transfer_entropy(X, Y, k=None, l=None, bins=8):
    """
    Compute transfer entropy T_{X->Y} (Equation 3).
    
    TE = sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) * log(p(y_{t+1}|y_t^{(k)}, x_t^{(l)}) / p(y_{t+1}|y_t^{(k)}))
    
    Parameters
    ----------
    X, Y : ndarray
        1D time series (X = source, Y = target).
    k, l : int
        History lengths for Y and X. If None, use adaptive_history_length.
    bins : int
        Number of bins for discretization.
        
    Returns
    -------
    te : float
        Transfer entropy value (bits).
    """
    if k is None:
        k, _ = adaptive_history_length(Y)
    if l is None:
        l, _ = adaptive_history_length(X)
    
    # Discretize signals
    X_disc = discretize_signal(X, bins)
    Y_disc = discretize_signal(Y, bins)
    
    n = len(X_disc)
    # Build joint distribution counts
    # We approximate using (y_{t+1}, y_t, x_t) triples for computational tractability
    te_sum = 0.0
    count_joint = {}
    count_yx = {}
    count_y = {}
    
    for t in range(max(k, l), n - 1):
        y_future = Y_disc[t + 1]
        y_past = tuple(Y_disc[t - k + 1:t + 1])
        x_past = tuple(X_disc[t - l + 1:t + 1])
        
        key_joint = (y_future, y_past, x_past)
        key_yx = (y_future, y_past)
        key_y = y_past
        
        count_joint[key_joint] = count_joint.get(key_joint, 0) + 1
        count_yx[key_yx] = count_yx.get(key_yx, 0) + 1
        count_y[key_y] = count_y.get(key_y, 0) + 1
    
    total = sum(count_joint.values())
    
    for key, cnt in count_joint.items():
        y_future, y_past, x_past = key
        key_yx = (y_future, y_past)
        key_y = y_past
        
        p_joint = cnt / total
        p_yx = count_yx.get(key_yx, 1) / total
        p_y = count_y.get(key_y, 1) / total
        
        if p_yx > 0 and p_y > 0:
            te_sum += p_joint * np.log2(p_yx / p_y)
    
    return max(0, te_sum)


def discretize_signal(signal, bins=8):
    """Uniform binning discretization for entropy estimation."""
    signal = np.asarray(signal)
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal, dtype=int)
    bins_edges = np.linspace(min_val, max_val, bins + 1)
    return np.digitize(signal, bins_edges[1:-1])


def build_directed_network(signals, channel_names=None, threshold=0.1):
    """
    Build directed brain network from multichannel EEG using TE.
    
    Parameters
    ----------
    signals : ndarray, shape (n_channels, n_samples)
    channel_names : list
    threshold : float
        TE threshold for edge inclusion.
        
    Returns
    -------
    adjacency : ndarray
        Weighted directed adjacency matrix.
    """
    n_ch = signals.shape[0]
    adjacency = np.zeros((n_ch, n_ch))
    
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            te_val = transfer_entropy(signals[i], signals[j])
            if te_val > threshold:
                adjacency[i, j] = te_val
    
    return adjacency


# === Example usage ===
if __name__ == '__main__':
    np.random.seed(42)
    
    # Simulate coupled signals
    t = np.linspace(0, 2, 500)
    X = np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(len(t))
    Y = np.sin(2 * np.pi * 5 * t + 0.5) + 0.3 * X[:-1].mean() + 0.3 * np.random.randn(len(t))
    Y = np.roll(Y, 1)  # Y lags X slightly
    
    # Adaptive history
    k, H = adaptive_history_length(Y, fs=250)
    print(f"Adaptive history length k={k}, Spectral entropy H={H:.3f}")
    
    # Transfer entropy
    te = transfer_entropy(X, Y, k=k, l=k)
    print(f"Transfer entropy T_{{X->Y}} = {te:.4f} bits")
    
    # Build network from 8-channel demo
    n_ch, n_samples = 8, 500
    demo_signals = np.random.randn(n_ch, n_samples)
    # Add coupling
    for i in range(1, n_ch):
        demo_signals[i] += 0.2 * demo_signals[i-1]
    
    adj = build_directed_network(demo_signals, threshold=0.05)
    print(f"Network density: {np.sum(adj > 0) / (n_ch * (n_ch - 1)):.2%}")
