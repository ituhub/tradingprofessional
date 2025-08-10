import os
import time
import threading
import logging
import streamlit as st
import traceback


class AppKeepAlive:
    """
    Enhanced keep-alive mechanism with improved thread management
    """
    _instance = None  # Singleton instance
    
    def __new__(cls, interval=600):
        """
        Ensure only one instance of AppKeepAlive is created
        """
        if not cls._instance:
            cls._instance = super(AppKeepAlive, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, interval=600):
        """
        Initialize keep-alive mechanism with thread-safe singleton pattern
        """
        # Prevent re-initialization if already set up
        if hasattr(self, '_initialized'):
            return
        
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self._initialized = True
        
        # Configure logging with unique handler to prevent duplicates
        self.logger = logging.getLogger('AppKeepAlive')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicate logging
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _keep_alive_signal(self):
        """
        Comprehensive keep-alive signal with error handling and diagnostics
        """
        # Use a thread-safe approach to update session state
        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()
        try:
            with self._lock:
                current_time = time.time()
                st.session_state.last_keep_alive = current_time

            # Lightweight background computation
            os_info = {
                'pid': os.getpid(),
                'timestamp': time.ctime()
            }

            self.logger.info("Keep-alive signal: %s", os_info)
        except (RuntimeError, KeyError) as e:
            self.logger.error("Keep-alive signal error: %s", e)
            self.logger.error("%s", traceback.format_exc())
    
    def start(self):
        """
        Start the keep-alive thread with additional safety checks
        """
        # Prevent multiple thread starts
        if self.thread and self.thread.is_alive():
            self.logger.warning("Keep-alive thread already running. Skipping restart.")
            return
        
        self.stop_event.clear()
        
        def run():
            """
            Background thread with enhanced error handling and logging
            """
            failure_count = 0
            max_failures = 5
            
            while not self.stop_event.is_set():
                try:
                    self._keep_alive_signal()
                    failure_count = 0  # Reset on successful execution
                    
                    # Dynamic wait with exponential backoff
                    wait_time = min(self.interval, 2 ** failure_count)
                    self.stop_event.wait(wait_time)
                
                except Exception as e:
                    failure_count += 1
                    self.logger.error(f"Keep-alive thread error (Attempt {failure_count}/{max_failures}): {e}")
                    
                    if failure_count >= max_failures:
                        self.logger.critical("Max keep-alive failures reached. Stopping thread.")
                        break
                    
                    # Exponential backoff
                    wait_time = min(2 ** failure_count, 300)  # Max 5 minutes
                    time.sleep(wait_time)
        
        # Use daemon thread to ensure it doesn't block application shutdown
        self.thread = threading.Thread(target=run, daemon=True, name='AppKeepAliveThread')
        self.thread.start()
        
        self.logger.info(f"Keep-alive thread started with interval {self.interval} seconds")
    
    def stop(self):
        """
        Safely stop the keep-alive thread
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
            'implementation': 'Hidden timer component in Streamlit'
        },
        {
            'name': 'Minimal Resource Approach',
            'description': 'Lightweight method to prevent app from sleeping',
            'implementation': 'Periodic logging or minimal computation'
        }
    ]
    
    return strategies
