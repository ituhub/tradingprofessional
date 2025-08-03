import streamlit as st
import time
import numpy as np
import pandas as pd
from functools import wraps
import tracemalloc

class MobilePerformanceOptimizer:
    """
    Performance optimization strategies for mobile and desktop devices
    """
    
    def __init__(self, is_mobile: bool = False):
        """
        Initialize performance optimizer
        
        Args:
            is_mobile (bool): Flag to indicate mobile device
        """
        self.is_mobile = is_mobile
        self.performance_config = {
            'max_data_size': 500 if is_mobile else 2000,
            'cache_timeout': 1800 if is_mobile else 3600,  # 30 min for mobile, 1 hour for desktop
            'performance_logging': is_mobile
        }
    
    def cache_mobile_data(self, timeout=None):
        """
        Decorator to cache data with mobile-optimized caching
        
        Args:
            timeout: Cache timeout in seconds
        """
        if timeout is None:
            timeout = self.performance_config['cache_timeout']
        
        def decorator(func):
            @wraps(func)
            @st.cache_data(ttl=timeout)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def performance_timer(self, func):
        """
        Decorator to log and track function performance
        
        Args:
            func: Function to be timed
        
        Returns:
            Wrapped function with performance tracking
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start memory tracking
            tracemalloc.start()
            
            # Time the function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Check memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Performance logging for mobile devices
            if self.is_mobile and self.performance_config['performance_logging']:
                if execution_time > 2.0:  # 2 seconds threshold
                    st.warning(
                        f"⏱️ Performance Warning:\n"
                        f"Function: {func.__name__}\n"
                        f"Execution Time: {execution_time:.2f}s\n"
                        f"Memory Usage: {current / 10**6:.2f} MB"
                    )
            
            return result
        return wrapper
    
    def lightweight_data_processing(self, data):
        """
        Reduce data size for mobile processing
        
        Args:
            data: Input data
        
        Returns:
            Lightweight processed data
        """
        max_size = self.performance_config['max_data_size']
        
        try:
            # Handle different data types
            if hasattr(data, 'head'):  # Pandas DataFrame
                return data.head(max_size)
            elif isinstance(data, list):
                return data[:max_size]
            elif isinstance(data, np.ndarray):
                return data[:max_size]
            
            return data
        
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return data
    
    def adaptive_data_sampling(self, data, sample_strategy='random'):
        """
        Intelligently sample data based on device type
        
        Args:
            data: Input data
            sample_strategy: Sampling method ('random', 'stratified')
        
        Returns:
            Sampled dataset
        """
        max_size = self.performance_config['max_data_size']
        
        try:
            if isinstance(data, pd.DataFrame):
                if sample_strategy == 'random':
                    # Random sampling
                    return data.sample(n=min(len(data), max_size))
                elif sample_strategy == 'stratified' and 'category' in data.columns:
                    # Stratified sampling if 'category' column exists
                    return data.groupby('category', group_keys=False).apply(
                        lambda x: x.sample(n=min(len(x), max_size // len(data['category'].unique())))
                    )
            elif isinstance(data, np.ndarray):
                # Random sampling for numpy arrays
                sample_indices = np.random.choice(
                    len(data), 
                    size=min(len(data), max_size), 
                    replace=False
                )
                return data[sample_indices]
            elif isinstance(data, list):
                # Simple list sampling
                return data[:max_size]
            
            return data
        
        except Exception as e:
            st.error(f"Data sampling error: {e}")
            return data
    
    def memory_efficient_processing(self, data, processing_func):
        """
        Wrap data processing function with memory efficiency
        
        Args:
            data: Input data
            processing_func: Function to process data
        
        Returns:
            Processed data
        """
        # Reduce data size before processing
        reduced_data = self.lightweight_data_processing(data)
        
        # Apply processing function
        return processing_func(reduced_data)
    
    def get_performance_report(self):
        """
        Generate performance report
        
        Returns:
            Dict with performance configuration
        """
        return {
            'is_mobile': self.is_mobile,
            'max_data_size': self.performance_config['max_data_size'],
            'cache_timeout': self.performance_config['cache_timeout'],
            'performance_logging': self.performance_config['performance_logging']
        }

# Create a global performance optimizer
def create_mobile_performance_optimizer(is_mobile: bool = False) -> MobilePerformanceOptimizer:
    """
    Create and return a mobile performance optimizer
    
    Args:
        is_mobile (bool): Flag to indicate mobile device
    
    Returns:
        MobilePerformanceOptimizer instance
    """
    return MobilePerformanceOptimizer(is_mobile)