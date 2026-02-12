"""Evaluation metrics"""
from typing import List, Dict

def calculate_asr(results: List[Dict]) -> float:
    """
    Calculate Attack Success Rate
    
    Args:
        results: List of result entries
        
    Returns:
        ASR value between 0-1
    """
    if not results:
        return 0.0
    
    successful = sum(1 for r in results if r.get('AS') == 1)
    return successful / len(results)