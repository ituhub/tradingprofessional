import streamlit as st
import numpy as np
from typing import Dict, Any, List

class MobileConfigManager:
    """
    Manage mobile-specific configurations and settings
    """
    
    def __init__(self, is_mobile: bool = False):
        """
        Initialize mobile configuration settings
        
        Args:
            is_mobile (bool): Flag to indicate mobile device
        """
        self.is_mobile = is_mobile
        self.mobile_config = {
            'reduced_feature_set': is_mobile,
            'compact_visualization': is_mobile,
            'data_loading_threshold': 500 if is_mobile else 1000,
            'chart_height': 300 if is_mobile else 500,
            'chart_width': None,  # Full width
            'max_tickers': 3 if is_mobile else 10,
            'performance_mode': True
        }
    
    def get_mobile_config(self) -> Dict[str, Any]:
        """
        Retrieve mobile configuration
        
        Returns:
            Dict containing mobile-specific settings
        """
        return self.mobile_config
    
    def apply_mobile_limits(self, data):
        """
        Apply data loading limits for mobile
        
        Args:
            data: Pandas DataFrame or similar data structure
        
        Returns:
            Reduced dataset suitable for mobile
        """
        max_threshold = self.mobile_config['data_loading_threshold']
        
        try:
            if hasattr(data, 'head'):  # Pandas DataFrame
                return data.head(max_threshold)
            elif isinstance(data, list):
                return data[:max_threshold]
            elif isinstance(data, np.ndarray):
                return data[:max_threshold]
            
            return data
        
        except Exception as e:
            st.error(f"Data limit application error: {e}")
            return data
    
    def mobile_ticker_limit(self, tickers: List[str]) -> List[str]:
        """
        Limit number of tickers for mobile
        
        Args:
            tickers: List of tickers
        
        Returns:
            Limited list of tickers
        """
        max_tickers = self.mobile_config['max_tickers']
        return tickers[:max_tickers]
    
    def configure_mobile_plotly(self, fig):
        """
        Configure Plotly figure for mobile
        
        Args:
            fig: Plotly figure object
        
        Returns:
            Modified Plotly figure
        """
        if self.mobile_config['compact_visualization']:
            fig.update_layout(
                height=self.mobile_config['chart_height'],
                width=self.mobile_config['chart_width'],
                margin=dict(l=10, r=10, t=30, b=10)
            )
        return fig
    
    def get_mobile_friendly_columns(self, total_columns: int = 4) -> List:
        """
        Generate mobile-friendly column layout
        
        Args:
            total_columns: Total number of columns to create
        
        Returns:
            Streamlit columns
        """
        # For mobile, limit to 2 columns max
        return st.columns(min(total_columns, 2))
    
    def apply_mobile_data_sampling(self, data):
        """
        Apply intelligent data sampling for mobile
        
        Args:
            data: Input data (DataFrame, numpy array, etc.)
        
        Returns:
            Sampled or reduced dataset
        """
        # If mobile, apply intelligent sampling
        if self.is_mobile:
            if hasattr(data, 'sample'):
                # For DataFrames, use stratified sampling
                sample_size = min(len(data), self.mobile_config['data_loading_threshold'])
                return data.sample(n=sample_size)
            elif isinstance(data, np.ndarray):
                # For numpy arrays, random sampling
                sample_indices = np.random.choice(
                    len(data), 
                    size=min(len(data), self.mobile_config['data_loading_threshold']), 
                    replace=False
                )
                return data[sample_indices]
        
        return data

# Create a global mobile config manager
def create_mobile_config_manager(is_mobile: bool = False) -> MobileConfigManager:
    """
    Create and return a mobile configuration manager
    
    Args:
        is_mobile (bool): Flag to indicate mobile device
    
    Returns:
        MobileConfigManager instance
    """
    return MobileConfigManager(is_mobile)