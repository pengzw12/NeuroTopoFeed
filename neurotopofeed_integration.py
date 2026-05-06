"""
neurotopofeed_integration.py
Complete System Integration for Real-Time Processing

Orchestrates PRISM-EEG, transfer entropy, incremental PH, and ATBCR dynamics
into a unified real-time BCI pipeline.

Dependencies: numpy, scipy, sklearn
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from prism_eeg import PRISMEEG
from transfer_entropy import build_directed_network, adaptive_history_length
from incremental_ph import IncrementalPersistentHomology, TopologicalBiomarkers
from atbcr_dynamics import ATBCRKuramotoDynamics


class NeuroTopoFeed:
    """
    Complete NeuroTopoFeed real-time system.
    
    Pipeline: Raw EEG -> PRISM-EEG -> TE Network -> Incremental PH ->
              Topological Features + ATBCR Dynamics -> Classification
    
    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    n_channels : int
        EEG channel count.
    window_sec : float
        Analysis window length.
    """
    
    def __init__(self, fs=250.0, n_channels=64, window_sec=2.0):
        self.fs = fs
        self.n_channels = n_channels
        self.window_sec = window_sec
        self.window_samples = int(fs * window_sec)
        
        # Components
        self.prism = PRISMEEG(fs=fs)
        self.iph = IncrementalPersistentHomology(n_nodes=n_channels)
        self.biomarkers = TopologicalBiomarkers()
        self.dynamics = None  # Initialized per session
        
        # Classification
        self.scaler = StandardScaler()
        self.classifier = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        self.is_trained = False
        
        # State tracking
        self.feature_buffer = []
        self.label_buffer = []
        self.current_state = None
    
    def extract_features(self, eeg_window):
        """
        Extract full feature vector from EEG window.
        
        Returns
        -------
        features : ndarray
            Concatenated topological + spectral + dynamic features.
        """
        # 1. PRISM-EEG spectral features
        prism_features = self.prism.transform(eeg_window)
        spectral_vec = prism_features.reshape(-1)  # flatten
        
        # 2. Build directed network via TE
        adjacency = build_directed_network(eeg_window, threshold=0.1)
        
        # 3. Incremental PH
        edges = []
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if adjacency[i, j] > 0:
                    edges.append((i, j, adjacency[i, j]))
        
        betti = self.iph.update(edges, current_time=0.0)  # time managed externally
        
        # 4. Topological biomarkers
        biomarker_vec = np.array([
            self.biomarkers.network_efficiency_index(betti),
            self.biomarkers.global_coherence_score(betti),
            self.biomarkers.hyperconnectivity_index(betti),
            self.biomarkers.sluggishness_index(betti),
        ])
        
        # 5. ATBCR dynamics (if initialized)
        dynamic_vec = np.zeros(4)
        if self.dynamics is not None:
            state_sig = self.dynamics.get_state_signature()
            dynamic_vec = np.array([
                state_sig['sync_index'],
                state_sig['mean_phase'] / (2 * np.pi),
                state_sig['phase_variance'],
                1.0 if state_sig['current_model'] == 'kuramoto' else 0.0,
            ])
        
        # Concatenate all features
        features = np.concatenate([spectral_vec, biomarker_vec, dynamic_vec])
        return features
    
    def fit(self, X_trials, y_labels):
        """
        Train the classification head on pre-extracted features.
        
        Parameters
        ----------
        X_trials : list of ndarray
            List of EEG trial windows.
        y_labels : list of str
            State labels.
        """
        feature_matrix = []
        for trial in X_trials:
            feat = self.extract_features(trial)
            feature_matrix.append(feat)
        
        X = np.array(feature_matrix)
        y = np.array(y_labels)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Train lightweight MLP
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        print(f"Training complete: {len(X_trials)} trials, {X.shape[1]} features")
    
    def predict(self, eeg_window):
        """
        Real-time prediction from single EEG window.
        
        Returns
        -------
        label : str
            Predicted state.
        prob : ndarray
            Class probabilities.
        biomarkers : dict
            Interpretable biomarkers.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        feat = self.extract_features(eeg_window)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        
        label = self.classifier.predict(feat_scaled)[0]
        prob = self.classifier.predict_proba(feat_scaled)[0]
        
        # Extract current biomarkers
        betti = self.iph.betti
        biomarkers = {
            'state': label,
            'confidence': np.max(prob),
            'network_efficiency': self.biomarkers.network_efficiency_index(betti),
            'global_coherence': self.biomarkers.global_coherence_score(betti),
            'hyperconnectivity': self.biomarkers.hyperconnectivity_index(betti),
            'beta_0': betti[0],
            'beta_1': betti[1],
            'beta_2': betti[2],
        }
        
        return label, prob, biomarkers
    
    def process_stream(self, eeg_stream, step_ms=500):
        """
        Process continuous EEG stream with sliding window.
        
        Parameters
        ----------
        eeg_stream : ndarray, shape (n_channels, n_samples)
            Continuous EEG.
        step_ms : int
            Step size in milliseconds.
            
        Returns
        -------
        predictions : list of dict
            Output at each window.
        """
        step_samples = int(self.fs * step_ms / 1000)
        n_total = eeg_stream.shape[1]
        
        predictions = []
        for start in range(0, n_total - self.window_samples + 1, step_samples):
            window = eeg_stream[:, start:start + self.window_samples]
            label, prob, bio = self.predict(window)
            predictions.append({
                'time': start / self.fs,
                'label': label,
                'confidence': float(np.max(prob)),
                **bio
            })
        
        return predictions
    
    def reset_session(self):
        """Reset all stateful components for new recording session."""
        self.iph.reset()
        self.current_state = None
        self.feature_buffer.clear()
        self.label_buffer.clear()
        if self.dynamics is not None:
            self.dynamics = None
    
    @property
    def latency_budget_ms(self):
        """Return component latency breakdown (Table 11)."""
        return {
            'PRISM-EEG': 42,
            'TE_Estimation': 28,
            'PH_Computation': 15,
            'ATBCR_Prediction': 3,
            'Total_Algorithm': 88,
            'Total_System': 156,
        }


# === Example usage ===
if __name__ == '__main__':
    np.random.seed(42)
    
    # Initialize system
    ntf = NeuroTopoFeed(fs=250.0, n_channels=64, window_sec=2.0)
    
    # Simulate training data: 4 states x 10 trials
    from jansen_rit import CoupledJansenRitNetwork
    
    network = CoupledJansenRitNetwork(n_nodes=64)
    X_train, y_train = network.generate_dataset(trials_per_state=10, duration=2.0)
    
    # Train
    ntf.fit(X_train, y_train)
    
    # Test on new trial
    test_signal, _ = network.simulate_state('Anxiety', duration=2.0)
    label, prob, bio = ntf.predict(test_signal)
    
    print(f"\nPrediction: {label} (confidence={bio['confidence']:.2f})")
    print(f"Biomarkers:")
    print(f"  Network Efficiency Index: {bio['network_efficiency']:.3f}")
    print(f"  Hyper-connectivity Index: {bio['hyperconnectivity']:.3f}")
    print(f"  Betti: [{bio['beta_0']}, {bio['beta_1']}, {bio['beta_2']}]")
    
    print(f"\nLatency Budget:")
    for comp, lat in ntf.latency_budget_ms.items():
        print(f"  {comp}: {lat} ms")
