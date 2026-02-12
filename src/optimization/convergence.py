"""Convergence detection for ASR"""
from typing import List

class ConvergenceDetector:
    """Detect ASR convergence"""
    
    def __init__(self, window: int = 3, epsilon: float = 0.01):
        """
        Initialize convergence detector
        
        Args:
            window: Number of iterations to check
            epsilon: Minimum improvement threshold
        """
        self.window = window
        self.epsilon = epsilon
        self.history = []
    
    def add(self, asr: float):
        """Add ASR value to history"""
        self.history.append(asr)
    
    def check(self, current_asr: float) -> bool:
        """
        Check if ASR has converged
        
        Args:
            current_asr: Current ASR value
            
        Returns:
            True if converged, False otherwise
        """
        self.add(current_asr)
        
        if len(self.history) < self.window:
            return False
        
        last_n = self.history[-self.window:]
        
        # Check if steadily increasing
        improvement = last_n[-1] - last_n[0]
        
        return improvement <= self.epsilon
    
    def reset(self):
        """Reset history"""
        self.history = []