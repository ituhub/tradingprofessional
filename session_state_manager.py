import streamlit as st
from datetime import datetime

def initialize_session_state():
    """
    Comprehensive session state initialization
    Ensures all necessary attributes are set with default values
    """
    # Initialize only if not already set
    if 'initialized' not in st.session_state:
        # Subscription and access management
        st.session_state.subscription_tier = 'free'
        st.session_state.premium_key = ''
        st.session_state.subscription_info = {}
        
        # Security state
        st.session_state.security_state = {
            'authenticated': False,
            'user': None,
            'login_attempts': 0,
            'last_login_attempt': None
        }
        
        # Selection and configuration
        st.session_state.selected_ticker = '^GSPC'  # Default ticker
        st.session_state.selected_timeframe = '1day'
        st.session_state.selected_models = []
        
        # Prediction and analysis results
        st.session_state.current_prediction = None
        st.session_state.real_ensemble_results = {}
        st.session_state.cross_validation_results = {}
        st.session_state.model_performance_metrics = {}
        st.session_state.forecast_data = []
        st.session_state.confidence_analysis = {}
        
        # Advanced analytics
        st.session_state.regime_analysis = {}
        st.session_state.real_risk_metrics = {}
        st.session_state.drift_detection_results = {}
        st.session_state.model_explanations = {}
        st.session_state.real_alternative_data = {}
        st.session_state.economic_indicators = {}
        st.session_state.sentiment_data = {}
        st.session_state.options_flow_data = {}
        
        # Backtesting and portfolio
        st.session_state.backtest_results = {}
        st.session_state.portfolio_optimization_results = {}
        st.session_state.strategy_performance = {}
        
        # Real-time data streams
        st.session_state.real_time_prices = {}
        st.session_state.hf_features = {}
        st.session_state.market_regime = None
        st.session_state.last_update = None
        st.session_state.market_status = {'isMarketOpen': True}
        
        # Model management
        st.session_state.models_trained = {}
        st.session_state.model_configs = {}
        st.session_state.training_history = {}
        
        # Usage tracking
        st.session_state.daily_usage = {
            'predictions': 0, 
            'date': datetime.now().date()
        }
        st.session_state.session_stats = {
            'predictions': 0,
            'models_trained': 0,
            'backtests': 0,
            'cv_runs': 0,
            'explanations_generated': 0
        }
        
        # Backend objects placeholders
        st.session_state.data_manager = None
        st.session_state.economic_provider = None
        st.session_state.sentiment_provider = None
        st.session_state.options_provider = None
        
        # Keepalive and system state
        st.session_state.last_keep_alive = datetime.now()
        
        # Mark as initialized
        st.session_state.initialized = True

def reset_session_state():
    """
    Reset all session state variables to their default values
    Useful for logout or complete reset
    """
    # Remove all existing session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Re-initialize
    initialize_session_state()

def update_session_state(updates: dict):
    """
    Safely update specific session state variables
    
    Args:
        updates (dict): Dictionary of session state updates
    """
    # Ensure session state is initialized
    initialize_session_state()
    
    # Update specified keys
    for key, value in updates.items():
        st.session_state[key] = value

# Export key functions
__all__ = [
    'initialize_session_state', 
    'reset_session_state', 
    'update_session_state'
]