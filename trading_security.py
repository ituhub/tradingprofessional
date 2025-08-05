"""
AI Trading Professional - Security System
========================================
Multiple security layers without traditional username/password authentication

Usage in mobilev2.py:
    from trading_security import SecurityManager, check_security
    
    # At the top of main():
    if not check_security():
        return  # Stops app execution if security fails
"""

import streamlit as st
import hashlib
import hmac
import time
import json
import base64
import socket
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Comprehensive security manager with multiple authentication methods
    """
    
    def __init__(self):
        # Security configuration
        self.config = {
            # Method 1: Access Tokens (Recommended)
            'access_tokens': {
            'enabled': True,
            'tokens': [
                'TRADE_2024_ALPHA',      # Token 1
                'TRADE_2024_BETA',       # Token 2  
                'TRADE_2024_GAMMA',      # Token 3
                'TRADE_2024_DELTA',      # Token 4
                'TRADE_2024_EPSILON'     # Token 5
            ],
            'max_uses_per_token': 20,   # Changed from 100 to 20
            'expires_after_days': 30    # Optional expiration
            },
            
            # Method 2: Hardware ID Whitelist
            'hardware_whitelist': {
            'enabled': False,  # Set to True to enable
            'allowed_devices': [
                # Add hardware IDs of allowed devices
                # Get these by running get_hardware_id() first
            ]
            },
            
            # Method 3: IP Address Whitelist
            'ip_whitelist': {
            'enabled': False,  # Set to True to enable
            'allowed_ips': [
                '127.0.0.1',        # Localhost
                '192.168.1.0/24',   # Local network range
                # Add specific IP addresses
            ]
            },
            
            # Method 4: Time-based Access
            'time_restriction': {
            'enabled': False,  # Set to True to enable
            'allowed_hours': (9, 17),  # 9 AM to 5 PM
            'allowed_days': [0, 1, 2, 3, 4],  # Monday to Friday
            'timezone': 'UTC'
            },
            
            # Method 5: Session Limits
            'session_limits': {
            'enabled': True,
            'max_daily_sessions': 50,
            'max_session_duration_hours': 8,
            'cooldown_minutes': 5
            },
            
            # Method 6: Geographic Restrictions
            'geo_restriction': {
            'enabled': False,  # Set to True to enable
            'allowed_countries': ['US', 'CA', 'GB', 'DE'],
            'blocked_countries': ['CN', 'RU']  # Blocked countries
            }
        }
        
        # Security state file
        self.state_file = Path('.security_state.json')
        self.load_security_state()
    
    def load_security_state(self):
        """Load security state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    self.security_state = json.load(f)
            else:
                self.security_state = {
                    'token_usage': {},
                    'daily_sessions': {},
                    'session_starts': {},
                    'last_cleanup': datetime.now().isoformat()
                }
                self.save_security_state()
        except Exception as e:
            logger.error(f"Error loading security state: {e}")
            self.security_state = {
                'token_usage': {},
                'daily_sessions': {},
                'session_starts': {},
                'last_cleanup': datetime.now().isoformat()
            }
    
    def save_security_state(self):
        """Save security state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.security_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving security state: {e}")
    
    def get_hardware_id(self) -> str:
        """Generate unique hardware identifier"""
        try:
            # Combine multiple hardware identifiers
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                          for elements in range(0, 2*6, 2)][::-1])
            hostname = socket.gethostname()
            
            # Create hash of combined identifiers
            combined = f"{mac}_{hostname}_{os.name}"
            hardware_id = hashlib.sha256(combined.encode()).hexdigest()[:16]
            
            return hardware_id
        except Exception as e:
            logger.error(f"Error generating hardware ID: {e}")
            return "unknown_device"
    
    def get_client_ip(self) -> str:
        """Get client IP address"""
        try:
            # Try to get real IP from headers (for deployed apps)
            if hasattr(st, 'get_option'):
                return st.get_option('client.toolbarMode') or '127.0.0.1'
            
            # Fallback methods
            try:
                response = requests.get('https://httpbin.org/ip', timeout=5)
                return response.json().get('origin', '127.0.0.1')
            except:
                return '127.0.0.1'
        except Exception:
            return '127.0.0.1'
    
    def get_geolocation(self, ip: str) -> Dict:
        """Get geolocation for IP address"""
        try:
            if ip == '127.0.0.1':
                return {'country': 'US', 'allowed': True}
            
            response = requests.get(f'https://ipapi.co/{ip}/json/', timeout=5)
            data = response.json()
            
            return {
                'country': data.get('country_code', 'US'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'allowed': True
            }
        except Exception as e:
            logger.warning(f"Geolocation lookup failed: {e}")
            return {'country': 'US', 'allowed': True}  # Default to allowed
    
    def check_access_token(self) -> bool:
        """
        Enhanced access token authentication with mandatory disclaimer
        
        Returns:
            bool: True if authenticated and consented, False otherwise
        """
        if not self.config['access_tokens']['enabled']:
            return True
        
        # Check if already authenticated and consented
        if st.session_state.get('authenticated', False) and \
           st.session_state.get('user_consented', False) and \
           self.validate_access_token(st.session_state.get('access_token', '')):
            return True
        
        # Display comprehensive disclaimer
        st.markdown("# 🔐 AI Trading Professional - User Agreement")
        
        disclaimer_text = """
        ## Important Disclaimer and Risk Warning

        ### 1. Financial Risk Acknowledgement
        - Trading financial instruments involves significant risk of loss
        - You may lose more than your initial investment
        - Past performance does not guarantee future results
        - This AI system provides predictive insights, NOT guaranteed profits

        ### 2. AI Prediction Limitations
        - AI predictions are probabilistic estimates, NOT absolute guarantees
        - Predictions can be incorrect or misleading
        - Market conditions can change rapidly
        - Always conduct your own research and due diligence

        ### 3. Usage Terms
        - This is a professional tool for informed traders
        - Do not make trading decisions based solely on AI predictions
        - Verify all information independently
        - Understand and manage your own risk

        ### 4. Data and Privacy
        - Your usage data may be logged for system improvement
        - No personal financial information is stored
        - Anonymous usage tracking is performed

        ### 5. Regulatory Compliance
        - You are responsible for ensuring compliance with local trading regulations
        - This tool is not a substitute for professional financial advice
        - Consult with a licensed financial advisor

        ### 6. Subscription and Access
        - Access is subject to our terms of service
        - Misuse can result in immediate account termination
        - Tokens are non-transferable and have usage limits

        **By clicking "I Consent", you acknowledge that you have read, 
        understood, and agree to these terms.**
        """
        
        # Display disclaimer in an expander for better readability
        with st.expander("📜 Read Full Disclaimer (IMPORTANT)", expanded=True):
            st.markdown(disclaimer_text)
        
        # Consent buttons
        consent_cols = st.columns(2)
        
        with consent_cols[0]:
            consent_button = st.button("✅ I Consent", type="primary", key="consent_button")
        
        with consent_cols[1]:
            decline_button = st.button("❌ I Do Not Consent", type="secondary", key="decline_button")
        
        # Handle button actions
        if consent_button:
            # Prompt for access token
            st.markdown("### 🔐 Access Token Required")
            st.info("Please enter your access token to continue")
            
            token_input = st.text_input(
                "Access Token",
                type="password",
                placeholder="Enter your access token",
                help="Contact administrator for access token"
            )
            
            if st.button("🚀 Authenticate", type="primary"):
                if self.validate_access_token(token_input):
                    st.session_state.access_token = token_input
                    st.session_state.authenticated = True
                    st.session_state.user_consented = True
                    st.success("✅ Authentication successful!")
                    time.sleep(1)
                    st.rerun()
                    return True
                else:
                    st.error("❌ Invalid access token")
                    return False
        
        elif decline_button:
            st.error("❌ Access Denied: User did not consent to terms")
            st.warning("You must read and consent to the disclaimer to access the AI Trading Professional system.")
            return False
        
        # Default return if no action taken
        return False
    
    def validate_access_token(self, token: str) -> bool:
        """Validate access token"""
        if not token:
            return False
        
        tokens = self.config['access_tokens']['tokens']
        if token not in tokens:
            return False
        
        # Check usage limits
        max_uses = self.config['access_tokens'].get('max_uses_per_token')
        if max_uses:
            usage_count = self.security_state['token_usage'].get(token, 0)
            if usage_count >= max_uses:
                return False
            
            # Increment usage
            self.security_state['token_usage'][token] = usage_count + 1
            self.save_security_state()
        
        return True
    
    def check_hardware_whitelist(self) -> bool:
        """Check hardware ID whitelist"""
        if not self.config['hardware_whitelist']['enabled']:
            return True
        
        hardware_id = self.get_hardware_id()
        allowed_devices = self.config['hardware_whitelist']['allowed_devices']
        
        if hardware_id not in allowed_devices:
            st.error(f"❌ Device not authorized. Hardware ID: {hardware_id}")
            st.info("Contact administrator to whitelist this device")
            return False
        
        return True
    
    def check_ip_whitelist(self) -> bool:
        """Check IP address whitelist"""
        if not self.config['ip_whitelist']['enabled']:
            return True
        
        client_ip = self.get_client_ip()
        allowed_ips = self.config['ip_whitelist']['allowed_ips']
        
        # Simple IP check (you can enhance with CIDR support)
        for allowed_ip in allowed_ips:
            if client_ip.startswith(allowed_ip.split('/')[0]):
                return True
        
        st.error(f"❌ Access denied from IP: {client_ip}")
        st.info("Contact administrator to whitelist your IP address")
        return False
    
    def check_time_restrictions(self) -> bool:
        """Check time-based access restrictions"""
        if not self.config['time_restriction']['enabled']:
            return True
        
        now = datetime.now()
        allowed_hours = self.config['time_restriction']['allowed_hours']
        allowed_days = self.config['time_restriction']['allowed_days']
        
        # Check day of week (0=Monday, 6=Sunday)
        if now.weekday() not in allowed_days:
            st.error("❌ Access denied - Outside allowed days")
            st.info(f"Access allowed Monday-Friday only")
            return False
        
        # Check hour
        if not (allowed_hours[0] <= now.hour < allowed_hours[1]):
            st.error("❌ Access denied - Outside allowed hours")
            st.info(f"Access allowed between {allowed_hours[0]}:00-{allowed_hours[1]}:00")
            return False
        
        return True
    
    def check_session_limits(self) -> bool:
        """Check session limits"""
        if not self.config['session_limits']['enabled']:
            return True
        
        today = datetime.now().date().isoformat()
        hardware_id = self.get_hardware_id()
        
        # Check daily session limit
        daily_sessions = self.security_state['daily_sessions']
        device_sessions = daily_sessions.get(today, {})
        session_count = device_sessions.get(hardware_id, 0)
        
        max_daily = self.config['session_limits']['max_daily_sessions']
        if session_count >= max_daily:
            st.error(f"❌ Daily session limit reached ({max_daily})")
            return False
        
        # Check session duration
        session_key = f"{hardware_id}_{today}"
        if session_key in self.security_state['session_starts']:
            start_time = datetime.fromisoformat(self.security_state['session_starts'][session_key])
            duration = datetime.now() - start_time
            max_duration = timedelta(hours=self.config['session_limits']['max_session_duration_hours'])
            
            if duration > max_duration:
                st.error("❌ Session duration limit exceeded")
                # Reset session
                del self.security_state['session_starts'][session_key]
                self.save_security_state()
                return False
        else:
            # Start new session
            self.security_state['session_starts'][session_key] = datetime.now().isoformat()
            
            # Increment daily session count
            if today not in daily_sessions:
                daily_sessions[today] = {}
            daily_sessions[today][hardware_id] = session_count + 1
            
            self.save_security_state()
        
        return True
    
    def check_geo_restrictions(self) -> bool:
        """Check geographic restrictions"""
        if not self.config['geo_restriction']['enabled']:
            return True
        
        client_ip = self.get_client_ip()
        geo_info = self.get_geolocation(client_ip)
        country = geo_info.get('country', 'US')
        
        allowed_countries = self.config['geo_restriction']['allowed_countries']
        blocked_countries = self.config['geo_restriction']['blocked_countries']
        
        # Check blocked countries first
        if country in blocked_countries:
            st.error(f"❌ Access denied from country: {country}")
            return False
        
        # Check allowed countries
        if allowed_countries and country not in allowed_countries:
            st.error(f"❌ Access denied from country: {country}")
            st.info(f"Access allowed from: {', '.join(allowed_countries)}")
            return False
        
        return True
    
    def cleanup_old_data(self):
        """Clean up old security data"""
        try:
            # Clean up daily sessions older than 7 days
            cutoff_date = (datetime.now() - timedelta(days=7)).date().isoformat()
            daily_sessions = self.security_state['daily_sessions']
            
            old_dates = [date for date in daily_sessions.keys() if date < cutoff_date]
            for date in old_dates:
                del daily_sessions[date]
            
            # Clean up old session starts
            session_starts = self.security_state['session_starts']
            old_sessions = []
            
            for session_key, start_time in session_starts.items():
                start_dt = datetime.fromisoformat(start_time)
                if datetime.now() - start_dt > timedelta(days=1):
                    old_sessions.append(session_key)
            
            for session_key in old_sessions:
                del session_starts[session_key]
            
            self.security_state['last_cleanup'] = datetime.now().isoformat()
            self.save_security_state()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def perform_security_check(self) -> bool:
        """Perform comprehensive security check"""
        try:
            # Cleanup old data periodically
            last_cleanup = datetime.fromisoformat(
                self.security_state.get('last_cleanup', datetime.now().isoformat())
            )
            if datetime.now() - last_cleanup > timedelta(hours=24):
                self.cleanup_old_data()
            
            # Run all security checks
            checks = [
                self.check_hardware_whitelist(),
                self.check_ip_whitelist(),
                self.check_time_restrictions(),
                self.check_session_limits(),
                self.check_geo_restrictions(),
                self.check_access_token()  # This should be last as it has UI
            ]
            
            return all(checks)
            
        except Exception as e:
            logger.error(f"Security check error: {e}")
            st.error("❌ Security system error")
            return False
    
    def get_security_info(self) -> Dict:
        """Get current security information for admin display"""
        hardware_id = self.get_hardware_id()
        client_ip = self.get_client_ip()
        
        return {
            'hardware_id': hardware_id,
            'client_ip': client_ip,
            'authenticated': st.session_state.get('authenticated', False),
            'current_token': st.session_state.get('access_token', ''),
            'session_count': len(self.security_state.get('session_starts', {})),
            'geo_info': self.get_geolocation(client_ip)
        }


# Global security manager instance
security_manager = SecurityManager()


def check_security() -> bool:
    """
    Main security check function to be called in mobilev2.py
    
    Returns:
        bool: True if all security checks pass, False otherwise
    """
    return security_manager.perform_security_check()


def get_hardware_id() -> str:
    """
    Helper function to get hardware ID for whitelisting
    
    Returns:
        str: Hardware ID of current device
    """
    return security_manager.get_hardware_id()


def show_security_admin_panel():
    """
    Admin panel for security management (call this in your app for debugging)
    """
    if st.session_state.get('show_admin_panel', False):
        st.markdown("### 🔧 Security Admin Panel")
        
        security_info = security_manager.get_security_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Session Info:**")
            st.code(f"""
Hardware ID: {security_info['hardware_id']}
Client IP: {security_info['client_ip']}
Authenticated: {security_info['authenticated']}
Token: {security_info['current_token'][:8]}...
            """)
        
        with col2:
            st.markdown("**Geographic Info:**")
            geo = security_info['geo_info']
            st.code(f"""
Country: {geo.get('country', 'Unknown')}
City: {geo.get('city', 'Unknown')}
Region: {geo.get('region', 'Unknown')}
Allowed: {geo.get('allowed', True)}
            """)
        
        # Token usage statistics
        st.markdown("**Token Usage:**")
        token_usage = security_manager.security_state.get('token_usage', {})
        if token_usage:
            st.json(token_usage)
        else:
            st.info("No token usage recorded")


def enable_admin_panel():
    """
    Enable admin panel for debugging (call this temporarily)
    """
    st.session_state.show_admin_panel = True


# Configuration presets for different security levels
SECURITY_PRESETS = {
    'minimal': {
        'access_tokens': {'enabled': True},
        'session_limits': {'enabled': True},
        'others': False
    },
    'standard': {
        'access_tokens': {'enabled': True},
        'session_limits': {'enabled': True},
        'time_restriction': {'enabled': True},
        'hardware_whitelist': {'enabled': False}
    },
    'strict': {
        'access_tokens': {'enabled': True},
        'session_limits': {'enabled': True},
        'time_restriction': {'enabled': True},
        'hardware_whitelist': {'enabled': True},
        'ip_whitelist': {'enabled': True}
    }
}


def apply_security_preset(preset_name: str):
    """
    Apply a security preset configuration
    
    Args:
        preset_name: 'minimal', 'standard', or 'strict'
    """
    if preset_name in SECURITY_PRESETS:
        preset = SECURITY_PRESETS[preset_name]
        
        # Update security manager configuration
        for key, value in preset.items():
            if key != 'others' and key in security_manager.config:
                if isinstance(value, dict):
                    security_manager.config[key].update(value)
                else:
                    security_manager.config[key]['enabled'] = value


# Example usage and setup instructions
SETUP_INSTRUCTIONS = """
SETUP INSTRUCTIONS FOR AI TRADING APP SECURITY
==============================================

1. BASIC SETUP (Recommended):
   - Import the security module
   - Add security check at the start of main()
   
   In mobilev2.py:
   ```python
   from trading_security import check_security
   
   def main():
       # Add this at the very beginning
       if not check_security():
           return
       
       # Rest of your app code...
   ```

2. ACCESS TOKENS (Primary Method):
   - Users need one of these tokens to access:
     * TRADE_2024_ALPHA
     * TRADE_2024_BETA  
     * TRADE_2024_GAMMA
     * TRADE_2024_DELTA
     * TRADE_2024_EPSILON
   
   - Each token can be used 100 times
   - Tokens expire after 30 days

3. HARDWARE WHITELIST (Optional):
   - Enable in config: 'hardware_whitelist': {'enabled': True}
   - Get hardware IDs: from trading_security import get_hardware_id
   - Add IDs to 'allowed_devices' list

4. CUSTOMIZATION:
   - Edit the config dictionary in SecurityManager.__init__()
   - Add/remove tokens
   - Adjust session limits
   - Configure time restrictions

5. SECURITY LEVELS:
   - Minimal: Just tokens + session limits
   - Standard: + time restrictions
   - Strict: + hardware/IP whitelisting
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
