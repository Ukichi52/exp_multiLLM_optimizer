"""Rate limiter for API calls"""
import time
from threading import Lock

class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize rate limiter
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()