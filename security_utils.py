import time
import logging
from functools import wraps
import streamlit as st

class RateLimiter:
    def __init__(self, max_calls=10, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = {}

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'user' not in st.session_state:
                st.error("Please log in first")
                return None

            user_id = st.session_state.user['id']
            current_time = time.time()

            # Clean up old calls
            self.calls[user_id] = [
                call_time for call_time in self.calls.get(user_id, []) 
                if current_time - call_time <= self.period
            ]

            # Check rate limit
            if len(self.calls.get(user_id, [])) >= self.max_calls:
                st.error(f"Rate limit exceeded. Try again in {self.period} seconds.")
                return None

            # Record call
            if user_id not in self.calls:
                self.calls[user_id] = []
            self.calls[user_id].append(current_time)

            return func(*args, **kwargs)
        return wrapper

class AuditLogger:
    def __init__(self, log_file='audit.log'):
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - User:%(user)s - %(message)s'
        )
        self.logger = logging.getLogger()

    def log_action(self, user, action, details=None):
        """Log user actions with context"""
        extra = {'user': user['username']}
        log_message = f"Action: {action}"
        if details:
            log_message += f" - Details: {details}"
        
        self.logger.info(log_message, extra=extra)

def validate_api_key(key):
    """Validate and sanitize API keys"""
    if not key or len(key) < 10:
        raise ValueError("Invalid API key")
    # Add additional validation specific to your API providers
    return key