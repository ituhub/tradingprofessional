import os
import time
import threading
import logging
import streamlit as st

class AppKeepAlive:
    """
    Simplified keep-alive mechanism without external dependencies
    """
    def __init__(self, interval=600):
        """
        Initialize keep-alive mechanism
        
        Args:
            interval (int, optional): Interval between keep-alive signals. Defaults to 10 minutes.
        """
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _keep_alive_signal(self):
        """
        Simulated keep-alive signal
        """
        try:
            # Simple in-memory state update to prevent idle state
            st.session_state.last_keep_alive = time.time()
            self.logger.info(f"Keep-alive signal at {time.ctime()}")
        except Exception as e:
            self.logger.error(f"Keep-alive signal error: {e}")
    
    def start(self):
        """
        Start the keep-alive thread
        """
        if self.thread and self.thread.is_alive():
            self.logger.warning("Keep-alive thread already running")
            return
        
        self.stop_event.clear()
        
        def run():
            """
            Background thread to periodically send keep-alive signals
            """
            while not self.stop_event.is_set():
                try:
                    self._keep_alive_signal()
                    # Wait for the specified interval
                    self.stop_event.wait(self.interval)
                except Exception as e:
                    self.logger.error(f"Error in keep-alive thread: {e}")
                    # Prevent tight loop in case of persistent errors
                    time.sleep(self.interval)
        
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        
        self.logger.info("Keep-alive thread started")
    
    def stop(self):
        """
        Stop the keep-alive thread
        """
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=5)
            self.logger.info("Keep-alive thread stopped")
    
    def __del__(self):
        """
        Ensure thread is stopped when object is deleted
        """
        self.stop()

def create_alternative_keep_alive_strategies():
    """
    Create alternative keep-alive strategies for different environments
    """
    strategies = [
        {
            'name': 'Session State Tracking',
            'description': 'Use Streamlit session state to track app activity',
            'implementation': 'Current AppKeepAlive mechanism'
        },
        {
            'name': 'Periodic UI Update',
            'description': 'Add a background timer to update UI periodically',
            'implementation': 'Create a hidden timer component in Streamlit'
        },
        {
            'name': 'Minimal Resource Approach',
            'description': 'Lightweight method to prevent app from sleeping',
            'implementation': 'Periodic logging or minimal computation'
        }
    ]
    
    return strategies


# Export key functions and classes

__all__ = [
    'AppKeepAlive', 
    'create_alternative_keep_alive_strategies'
]