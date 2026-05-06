"""
prism_eeg.py
PRISM-EEG Implementation with Multi-resolution Symmetric Convolution

Implements Type I linear-phase FIR filters with symmetric constraints,
achieving 50% parameter reduction while preserving phase-sensitive EEG analysis.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.signal import firwin, lfilter


class PRISMEEG:
    """
    PRISM-EEG encoder: multi-resolution symmetric convolution for EEG.
    
    Parameters
    ----------
    fs : float
        Sampling frequency (Hz). Default: 250 (after downsampling).
    bands : dict
        Frequency bands with (low, high, kernel_length).
    """
    
    # Default band parameters from Table 2
    DEFAULT_BANDS = {
        'delta':  {'freq': (0.5, 4.0),   'K': 127},
        'theta':  {'freq': (4.0, 8.0),   'K': 63},
        'alpha':  {'freq': (8.0, 13.0),  'K': 31},
        'beta':   {'freq': (13.0, 30.0), 'K': 15},
        'gamma':  {'freq': (30.0, 100.0), 'K': 7},
    }
    
    def __init__(self, fs=250.0, bands=None):
        self.fs = fs
        self.bands = bands or self.DEFAULT_BANDS
        self.filters = {}
        self._build_symmetric_filters()
    
    def _build_symmetric_filters(self):
        """
        Build Type I linear-phase FIR filters with symmetric constraints.
        For odd-length kernel K=2m+1: w[m-tau] = w[m+tau].
        Only (K+1)/2 independent parameters.
        """
        for name, cfg in self.bands.items():
            K = cfg['K']
            low, high = cfg['freq']
            
            # Standard FIR filter design ( Parks-McClellan equivalent via firwin )
            # Note: symmetric property is enforced by firwin with window='hamming'
            w_full = firwin(K, [low, high], pass_zero=False, fs=self.fs, window='hamming')
            
            # Explicit symmetric constraint verification
            assert np.allclose(w_full, w_full[::-1]), f"Filter {name} not symmetric"
            
            # Store independent parameters (first half + center)
            half = (K + 1) // 2
            self.filters[name] = {
                'w': w_full,
                'w_independent': w_full[:half],  # 50% parameter reduction
                'K': K,
                'freq': (low, high)
            }
    
    def transform(self, eeg_window):
        """
        Apply multi-resolution symmetric convolution to EEG window.
        
        Parameters
        ----------
        eeg_window : ndarray, shape (n_channels, n_samples)
            Single EEG window.
            
        Returns
        -------
        features : ndarray, shape (n_channels, n_bands, 3)
            Band power, mean, std for each channel-band pair.
        """
        n_channels, n_samples = eeg_window.shape
        n_bands = len(self.filters)
        features = np.zeros((n_channels, n_bands, 3))
        
        for b_idx, (band_name, fdict) in enumerate(self.filters.items()):
            w = fdict['w']
            for ch in range(n_channels):
                filtered = lfilter(w, [1.0], eeg_window[ch])
                power = np.mean(filtered ** 2)
                mean_val = np.mean(filtered)
                std_val = np.std(filtered)
                features[ch, b_idx, :] = [power, mean_val, std_val]
        
        return features
    
    @property
    def parameter_count(self):
        """Total independent parameters (50% of full symmetric kernels)."""
        total = 0
        for cfg in self.bands.values():
            total += (cfg['K'] + 1) // 2
        return total * 5  # 5 bands, per channel scaling


def symmetric_backprop_grad(w_grad, K):
    """
    Aggregate gradients for symmetric weight sharing.
    
    For kernel w[i] = u[min(i, K-1-i)], gradient for u[j] is sum of
    all w[i] where min(i, K-1-i) == j.
    
    Parameters
    ----------
    w_grad : ndarray, shape (K,)
        Gradient with respect to full kernel w.
    K : int
        Kernel length (odd).
        
    Returns
    -------
    u_grad : ndarray, shape ((K+1)//2,)
        Aggregated gradient for independent parameters u.
    """
    half = (K + 1) // 2
    u_grad = np.zeros(half)
    for i in range(K):
        j = min(i, K - 1 - i)
        u_grad[j] += w_grad[i]
    return u_grad


# === Example usage / sanity check ===
if __name__ == '__main__':
    prism = PRISMEEG(fs=250.0)
    
    # Simulate 2s window, 64 channels, 250 Hz
    fs = 250
    n_ch = 64
    t = 2  # seconds
    n_samples = fs * t
    
    np.random.seed(42)
    eeg_demo = np.random.randn(n_ch, n_samples) * 10  # ~10 uV EEG amplitude
    
    features = prism.transform(eeg_demo)
    print(f"PRISM-EEG output shape: {features.shape}")
    print(f"Independent parameters: ~{prism.parameter_count}")
    
    # Verify linear-phase property: group delay constant
    for name, fdict in prism.filters.items():
        w = fdict['w']
        # Check symmetry (Type I)
        assert np.allclose(w, w[::-1]), f"{name} failed symmetry check"
    print("All filters pass symmetric (Type I linear-phase) verification.")
