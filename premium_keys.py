import streamlit as st
import uuid
import json
import hashlib
import logging
import os
import sqlite3
import pandas as pd
import requests
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

class DatabaseManager:
    """
    Manages database connections and basic database operations
    """
    def __init__(self, db_path='admin_management.db'):
        """
        Initialize the database manager
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        with self._get_connection() as conn:
            # Admin users table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_locked BOOLEAN DEFAULT FALSE,
                last_login DATETIME,
                failed_attempts INTEGER DEFAULT 0,
                lockout_time DATETIME
            )
            ''')
            
            # Enhanced user IDs table with tier tracking
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_ids (
                user_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                max_predictions INTEGER NOT NULL,
                predictions_used INTEGER DEFAULT 0,
                predictions_remaining INTEGER,
                premium_key TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                reset_date DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
            ''')
            
            # Premium keys table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS premium_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT,
                tier TEXT,
                max_predictions INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                is_used BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES user_ids(user_id)
            )
            ''')
            
            # Usage tracking table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES user_ids(user_id)
            )
            ''')
            
            # Admin logs table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS admin_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_username TEXT,
                action TEXT,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT
            )
            ''')
    
    def _get_connection(self):
        """
        Get a connection to the SQLite database
        
        Returns:
            sqlite3.Connection: Database connection
        """
        return sqlite3.connect(self.db_path)

class EnhancedUserManager:
    """
    Enhanced user management with tier-based system
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tier_configs = {
            'free': {'max_predictions': 0, 'display_name': 'Free Tier'},
            'tier_10': {'max_predictions': 10, 'display_name': '10 Predictions'},
            'tier_25': {'max_predictions': 25, 'display_name': '25 Predictions'},
            'tier_50': {'max_predictions': 50, 'display_name': '50 Predictions'},
            'tier_100': {'max_predictions': 100, 'display_name': '100 Predictions'}
        }
    
    def generate_user_id_with_tier(self, tier: str, count: int = 1) -> List[Dict[str, str]]:
        """
        Generate user IDs with specified tier
        
        Args:
            tier (str): User tier
            count (int): Number of user IDs to generate
        
        Returns:
            List[Dict[str, str]]: Generated user IDs with details
        """
        if tier not in self.tier_configs:
            raise ValueError(f"Invalid tier: {tier}")
        
        tier_config = self.tier_configs[tier]
        generated_users = []
        
        with self.db_manager._get_connection() as conn:
            for _ in range(count):
                # Generate unique user ID
                user_id = f"USER-{uuid.uuid4().hex[:8].upper()}"
                
                # Generate premium key if not free tier
                premium_key = None
                if tier != 'free':
                    premium_key = f"PREM-{tier.upper()}-{uuid.uuid4().hex[:12].upper()}"
                
                # Calculate reset date (30 days from now)
                reset_date = datetime.now() + timedelta(days=30)
                max_predictions = tier_config['max_predictions']
                
                # Insert user into database
                conn.execute('''
                INSERT INTO user_ids 
                (user_id, tier, max_predictions, predictions_used, 
                 predictions_remaining, premium_key, reset_date, is_active) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, tier, max_predictions, 0, 
                    max_predictions, premium_key, reset_date.isoformat(), True
                ))
                
                # If premium key generated, add to premium_keys table
                if premium_key:
                    key_id = f"KEY-{uuid.uuid4().hex[:8].upper()}"
                    expires_at = datetime.now() + timedelta(days=365)  # 1 year expiry
                    
                    conn.execute('''
                    INSERT INTO premium_keys 
                    (key_id, user_id, tier, max_predictions, expires_at, is_used) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        key_id, user_id, tier, max_predictions, 
                        expires_at.isoformat(), False
                    ))
                
                generated_users.append({
                    'user_id': user_id,
                    'tier': tier,
                    'tier_display': tier_config['display_name'],
                    'max_predictions': max_predictions,
                    'premium_key': premium_key,
                    'reset_date': reset_date.strftime('%Y-%m-%d')
                })
        
        return generated_users
    
    def get_user_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a user
        
        Args:
            user_id (str): User ID to lookup
        
        Returns:
            Optional[Dict[str, Any]]: User details or None if not found
        """
        with self.db_manager._get_connection() as conn:
            cursor = conn.execute('''
            SELECT user_id, tier, max_predictions, predictions_used, 
                   predictions_remaining, premium_key, created_at, 
                   last_used, reset_date, is_active
            FROM user_ids 
            WHERE user_id = ?
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            
            if user_data:
                return {
                    'user_id': user_data[0],
                    'tier': user_data[1],
                    'tier_display': self.tier_configs[user_data[1]]['display_name'],
                    'max_predictions': user_data[2],
                    'predictions_used': user_data[3],
                    'predictions_remaining': user_data[4],
                    'premium_key': user_data[5],
                    'created_at': user_data[6],
                    'last_used': user_data[7],
                    'reset_date': user_data[8],
                    'is_active': user_data[9]
                }
            
            return None
    
    def use_prediction(self, user_id: str) -> bool:
        """
        Use a prediction for a user (decrement remaining predictions)
        
        Args:
            user_id (str): User ID
        
        Returns:
            bool: True if prediction was successfully used, False otherwise
        """
        with self.db_manager._get_connection() as conn:
            # Get current user status
            cursor = conn.execute('''
            SELECT predictions_remaining, is_active 
            FROM user_ids 
            WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result or not result[1]:  # User not found or inactive
                return False
            
            predictions_remaining = result[0]
            
            if predictions_remaining <= 0:
                return False
            
            # Update predictions
            conn.execute('''
            UPDATE user_ids 
            SET predictions_used = predictions_used + 1,
                predictions_remaining = predictions_remaining - 1,
                last_used = ?
            WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            
            # Log usage
            conn.execute('''
            INSERT INTO usage_tracking (user_id, action_type, details)
            VALUES (?, ?, ?)
            ''', (user_id, 'prediction_used', 'Prediction count decremented'))
            
            return True
    
    def reset_user_predictions(self, user_id: str) -> bool:
        """
        Reset user predictions to maximum (admin function)
        
        Args:
            user_id (str): User ID
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        with self.db_manager._get_connection() as conn:
            cursor = conn.execute('''
            SELECT max_predictions FROM user_ids WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return False
            
            max_predictions = result[0]
            new_reset_date = datetime.now() + timedelta(days=30)
            
            conn.execute('''
            UPDATE user_ids 
            SET predictions_used = 0,
                predictions_remaining = ?,
                reset_date = ?
            WHERE user_id = ?
            ''', (max_predictions, new_reset_date.isoformat(), user_id))
            
            # Log reset
            conn.execute('''
            INSERT INTO usage_tracking (user_id, action_type, details)
            VALUES (?, ?, ?)
            ''', (user_id, 'predictions_reset', f'Predictions reset to {max_predictions}'))
            
            return True
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users for admin dashboard
        
        Returns:
            List[Dict[str, Any]]: List of all users
        """
        with self.db_manager._get_connection() as conn:
            cursor = conn.execute('''
            SELECT user_id, tier, max_predictions, predictions_used, 
                   predictions_remaining, premium_key, created_at, 
                   last_used, reset_date, is_active
            FROM user_ids 
            ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'user_id': row[0],
                    'tier': row[1],
                    'tier_display': self.tier_configs[row[1]]['display_name'],
                    'max_predictions': row[2],
                    'predictions_used': row[3],
                    'predictions_remaining': row[4],
                    'premium_key': row[5],
                    'created_at': row[6],
                    'last_used': row[7],
                    'reset_date': row[8],
                    'is_active': row[9]
                })
            
            return users

class SecureAdminManager:
    """
    Manages secure admin authentication and related operations
    """
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the secure admin manager
        
        Args:
            db_manager (DatabaseManager): Database management instance
        """
        self.db_manager = db_manager
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """
        Set up logging for admin actions
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('admin_management')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'admin_actions.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Hash a password with a salt
        
        Args:
            password (str): Password to hash
            salt (Optional[str]): Salt to use (generate if not provided)
        
        Returns:
            Dict[str, str]: Dictionary with salt and password hash
        """
        if salt is None:
            salt = uuid.uuid4().hex
        
        # Use PBKDF2 with SHA-256 for password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        ).hex()
        
        return {
            'salt': salt,
            'password_hash': password_hash
        }
    
    def validate_login(self, username: str, password: str, ip_address: str) -> Dict[str, Any]:
        """
        Validate admin login with security checks
        
        Args:
            username (str): Admin username
            password (str): Admin password
            ip_address (str): Client IP address
        
        Returns:
            Dict[str, Any]: Login validation result
        """
        # Basic input validation
        if not username or not password:
            return {'valid': False, 'message': 'Username and password are required'}
        
        with self.db_manager._get_connection() as conn:
            # Fetch admin user details
            cursor = conn.execute('''
            SELECT username, password_hash, salt, role, 
                   is_active, is_locked, failed_attempts, lockout_time
            FROM admin_users 
            WHERE username = ?
            ''', (username,))
            
            admin_user = cursor.fetchone()
            
            # User not found
            if not admin_user:
                self.logger.warning(f"Login attempt for non-existent user: {username}")
                return {'valid': False, 'message': 'Invalid credentials'}
            
            # Unpack user details
            (db_username, db_password_hash, db_salt, db_role, 
             is_active, is_locked, failed_attempts, lockout_time) = admin_user
            
            # Check account status
            if not is_active:
                self.logger.warning(f"Login attempt for inactive user: {username}")
                return {'valid': False, 'message': 'Account is inactive'}
            
            # Check account lockout
            if is_locked:
                # Check if lockout period has expired
                if lockout_time and datetime.fromisoformat(lockout_time) > datetime.now():
                    self.logger.warning(f"Login attempt for locked user: {username}")
                    return {
                        'valid': False, 
                        'message': 'Account is temporarily locked. Try again later.'
                    }
            
            # Verify password
            hashed_input = self._hash_password(password, db_salt)
            if hashed_input['password_hash'] != db_password_hash:
                # Increment failed attempts
                new_failed_attempts = (failed_attempts or 0) + 1
                
                # Lock account after 5 failed attempts
                if new_failed_attempts >= 5:
                    lockout_time = datetime.now() + timedelta(minutes=15)
                    conn.execute('''
                    UPDATE admin_users 
                    SET failed_attempts = ?, is_locked = 1, lockout_time = ?
                    WHERE username = ?
                    ''', (new_failed_attempts, lockout_time.isoformat(), username))
                    
                    self.logger.warning(f"User {username} locked due to multiple failed login attempts")
                    return {
                        'valid': False, 
                        'message': 'Too many failed attempts. Account locked for 15 minutes.'
                    }
                else:
                    conn.execute('''
                    UPDATE admin_users 
                    SET failed_attempts = ?
                    WHERE username = ?
                    ''', (new_failed_attempts, username))
                
                self.logger.warning(f"Failed login attempt for user: {username}")
                return {'valid': False, 'message': 'Invalid credentials'}
            
            # Successful login - reset failed attempts
            conn.execute('''
            UPDATE admin_users 
            SET failed_attempts = 0, last_login = ?, is_locked = 0, lockout_time = NULL
            WHERE username = ?
            ''', (datetime.now().isoformat(), username))
            
            # Log successful login
            self.logger.info(f"Successful login for user: {username} from IP: {ip_address}")
            
            return {
                'valid': True, 
                'message': 'Login successful', 
                'role': db_role
            }
    
    def create_admin_user(self, username: str, password: str, role: str = 'admin'):
        """
        Create a new admin user
        
        Args:
            username (str): Admin username
            password (str): Admin password
            role (str): Admin role
        
        Raises:
            ValueError: If username already exists or input is invalid
        """
        # Input validation
        if not username or not password:
            raise ValueError("Username and password are required")
        
        # Hash password
        hashed_password = self._hash_password(password)
        
        with self.db_manager._get_connection() as conn:
            try:
                # Check if username already exists
                cursor = conn.execute('SELECT COUNT(*) FROM admin_users WHERE username = ?', (username,))
                if cursor.fetchone()[0] > 0:
                    raise ValueError("Username already exists")
                
                # Insert new admin user
                conn.execute('''
                INSERT INTO admin_users 
                (username, password_hash, salt, role, is_active, is_locked) 
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    username, 
                    hashed_password['password_hash'], 
                    hashed_password['salt'], 
                    role, 
                    True, 
                    False
                ))
                
                # Log admin user creation
                self.logger.info(f"New admin user created: {username} with role {role}")
            except sqlite3.IntegrityError:
                raise ValueError("Failed to create admin user")

class AdminDashboardApp:
    """
    Main admin dashboard application
    """
    def __init__(self):
        """Initialize the admin dashboard application"""
        self.db_manager = DatabaseManager()
        self.admin_manager = SecureAdminManager(self.db_manager)
        self.user_manager = EnhancedUserManager(self.db_manager)
        self._initialize_session_state()
        
        # Initialize default admin if none exists
        self._initialize_default_admin()
    
    def _initialize_session_state(self):
        """Initialize session state for admin functions"""
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
            st.session_state.admin_username = None
            st.session_state.admin_role = None
    
    def _initialize_default_admin(self):
        """Create default admin user if none exists"""
        with self.db_manager._get_connection() as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM admin_users')
            admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            try:
                # Create initial admin
                self.admin_manager.create_admin_user(
                    username='admin', 
                    password='admin123', 
                    role='admin'
                )
                st.info("ℹ️ Default admin created: username='admin', password='admin123'")
            except Exception as e:
                st.error(f"Error creating default admin: {e}")
    
    def login_page(self):
        """Admin login page with enhanced security"""
        st.title("🔐 Admin Login - User Management System")
        st.markdown("---")
        
        # Get client IP address
        try:
            ip_address = '127.0.0.1'  # Default for local development
        except Exception:
            ip_address = '127.0.0.1'
        
        # Login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔑 Administrator Access")
            username = st.text_input("👤 Username", placeholder="Enter admin username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if username and password:
                    login_result = self.admin_manager.validate_login(username, password, ip_address)
                    
                    if login_result['valid']:
                        st.session_state.admin_logged_in = True
                        st.session_state.admin_username = username
                        st.session_state.admin_role = login_result['role']
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {login_result['message']}")
                else:
                    st.error("❌ Please enter both username and password")
        
        # Information section
        st.markdown("---")
        st.info("""
        **📋 Default Credentials (for initial setup):**
        - Username: `admin`
        - Password: `admin123`
        
        **⚠️ Security Notice:** Please change the default password after first login.
        """)
    
    def dashboard_header(self):
        """Create dashboard header"""
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.title("🎯 Admin Dashboard")
        
        with col2:
            st.markdown(f"**👤 Welcome:** {st.session_state.admin_username}")
            st.markdown(f"**🔰 Role:** {st.session_state.admin_role}")
        
        with col3:
            if st.button("🚪 Logout", type="secondary"):
                self.logout()
    
    def user_generation_section(self):
        """Enhanced user generation section with two separate buttons"""
        st.header("👥 User ID & Premium Key Generation")
        st.markdown("---")
        
        # Tier selection
        tier_options = {
            'tier_10': '10 Predictions Tier',
            'tier_25': '25 Predictions Tier', 
            'tier_50': '50 Predictions Tier',
            'tier_100': '100 Predictions Tier'
        }
        
        free_tier_option = {'free': 'Free Tier (0 Predictions)'}
        
        col1, col2 = st.columns(2)
        
        # Premium Tiers Section
        with col1:
            st.subheader("🎯 Premium Tier Generation")
            st.markdown("*Generate User IDs with Premium Keys*")
            
            # Premium tier selection
            premium_tier = st.selectbox(
                "Select Premium Tier",
                options=list(tier_options.keys()),
                format_func=lambda x: tier_options[x],
                key="premium_tier_select"
            )
            
            # Number of IDs for premium
            premium_count = st.number_input(
                "Number of Premium User IDs",
                min_value=1,
                max_value=50,
                value=1,
                key="premium_count"
            )
            
            # Generate Premium IDs button
            if st.button("🎖️ Generate Premium User IDs", type="primary", key="generate_premium"):
                try:
                    generated_users = self.user_manager.generate_user_id_with_tier(premium_tier, premium_count)
                    
                    st.success(f"✅ Generated {len(generated_users)} Premium User IDs!")
                    
                    # Display generated users
                    for user in generated_users:
                        with st.expander(f"📋 {user['user_id']} - {user['tier_display']}", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.code(f"User ID: {user['user_id']}")
                                st.code(f"Tier: {user['tier_display']}")
                                st.code(f"Max Predictions: {user['max_predictions']}")
                            with col_b:
                                st.code(f"Premium Key: {user['premium_key']}")
                                st.code(f"Reset Date: {user['reset_date']}")
                    
                    # Option to download as CSV
                    if st.checkbox("💾 Download as CSV", key="download_premium"):
                        df = pd.DataFrame(generated_users)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Premium Users CSV",
                            data=csv,
                            file_name=f"premium_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating premium users: {e}")
        
        # Free Tier Section
        with col2:
            st.subheader("🆓 Free Tier Generation")
            st.markdown("*Generate User IDs for Free Tier (Tracking Only)*")
            
            # Number of IDs for free tier
            free_count = st.number_input(
                "Number of Free Tier User IDs",
                min_value=1,
                max_value=50,
                value=1,
                key="free_count"
            )
            
            st.info("📌 **Note:** Free tier users get 0 predictions but are tracked for usage analytics.")
            
            # Generate Free IDs button
            if st.button("🆓 Generate Free User IDs", type="secondary", key="generate_free"):
                try:
                    generated_users = self.user_manager.generate_user_id_with_tier('free', free_count)
                    
                    st.success(f"✅ Generated {len(generated_users)} Free Tier User IDs!")
                    
                    # Display generated users
                    for user in generated_users:
                        with st.expander(f"📋 {user['user_id']} - {user['tier_display']}", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.code(f"User ID: {user['user_id']}")
                                st.code(f"Tier: {user['tier_display']}")
                            with col_b:
                                st.code(f"Max Predictions: {user['max_predictions']}")
                                st.code(f"Reset Date: {user['reset_date']}")
                    
                    # Option to download as CSV
                    if st.checkbox("💾 Download as CSV", key="download_free"):
                        df = pd.DataFrame(generated_users)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Free Users CSV",
                            data=csv,
                            file_name=f"free_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating free users: {e}")
    
    def user_management_section(self):
        """Comprehensive user management and tracking"""
        st.header("📊 User Management & Tracking")
        st.markdown("---")
        
        # Get all users
        all_users = self.user_manager.get_all_users()
        
        if not all_users:
            st.info("📭 No users found. Generate some user IDs first.")
            return
        
        # Statistics overview
        st.subheader("📈 Usage Statistics Overview")
        
        # Calculate statistics
        total_users = len(all_users)
        active_users = len([u for u in all_users if u['is_active']])
        total_predictions_used = sum(u['predictions_used'] for u in all_users)
        total_predictions_available = sum(u['max_predictions'] for u in all_users)
        
        # Tier breakdown
        tier_breakdown = {}
        for user in all_users:
            tier_display = user['tier_display']
            tier_breakdown[tier_display] = tier_breakdown.get(tier_display, 0) + 1
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Users", total_users)
        with col2:
            st.metric("✅ Active Users", active_users)
        with col3:
            st.metric("🎯 Predictions Used", total_predictions_used)
        with col4:
            st.metric("📊 Total Available", total_predictions_available)
        
        # Tier breakdown
        st.subheader("🎭 User Distribution by Tier")
        tier_cols = st.columns(len(tier_breakdown))
        
        for i, (tier_name, count) in enumerate(tier_breakdown.items()):
            with tier_cols[i]:
                st.metric(tier_name, count)
        
        # User search and filtering
        st.subheader("🔍 Search & Filter Users")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_user_id = st.text_input("🔎 Search by User ID", placeholder="USER-XXXXXXXX")
        
        with col2:
            filter_tier = st.selectbox(
                "📊 Filter by Tier",
                options=["All"] + list(self.user_manager.tier_configs.keys()),
                format_func=lambda x: "All Tiers" if x == "All" else self.user_manager.tier_configs[x]['display_name']
            )
        
        with col3:
            filter_active = st.selectbox(
                "🔄 Filter by Status",
                options=["All", "Active", "Inactive"]
            )
        
        # Apply filters
        filtered_users = all_users.copy()
        
        if search_user_id:
            filtered_users = [u for u in filtered_users if search_user_id.upper() in u['user_id'].upper()]
        
        if filter_tier != "All":
            filtered_users = [u for u in filtered_users if u['tier'] == filter_tier]
        
        if filter_active != "All":
            is_active = filter_active == "Active"
            filtered_users = [u for u in filtered_users if u['is_active'] == is_active]
        
        # Display filtered users
        st.subheader(f"👥 User List ({len(filtered_users)} users)")
        
        if filtered_users:
            # Create DataFrame for better display
            display_data = []
            for user in filtered_users:
                display_data.append({
                    'User ID': user['user_id'],
                    'Tier': user['tier_display'],
                    'Used': user['predictions_used'],
                    'Remaining': user['predictions_remaining'],
                    'Max': user['max_predictions'],
                    'Premium Key': user['premium_key'][:20] + "..." if user['premium_key'] else "N/A",
                    'Created': user['created_at'][:10] if user['created_at'] else "N/A",
                    'Last Used': user['last_used'][:10] if user['last_used'] else "Never",
                    'Status': "✅ Active" if user['is_active'] else "❌ Inactive"
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
            
            # Bulk operations
            st.subheader("⚡ Bulk Operations")
            bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
            
            with bulk_col1:
                if st.button("🔄 Reset All Predictions", type="secondary"):
                    success_count = 0
                    for user in filtered_users:
                        if self.user_manager.reset_user_predictions(user['user_id']):
                            success_count += 1
                    st.success(f"✅ Reset predictions for {success_count} users")
                    st.rerun()
            
            with bulk_col2:
                if st.button("📥 Export to CSV", type="secondary"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Users CSV",
                        data=csv,
                        file_name=f"user_management_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with bulk_col3:
                if st.button("📊 Usage Report", type="secondary"):
                    self.generate_usage_report(filtered_users)
        
        else:
            st.info("📭 No users match the current filters.")
    
    def generate_usage_report(self, users):
        """Generate detailed usage report"""
        st.subheader("📈 Detailed Usage Report")
        
        # Usage analytics
        total_users = len(users)
        active_users = len([u for u in users if u['predictions_used'] > 0])
        usage_rate = (active_users / total_users * 100) if total_users > 0 else 0
        
        # Predictions analytics
        total_allocated = sum(u['max_predictions'] for u in users)
        total_used = sum(u['predictions_used'] for u in users)
        usage_percentage = (total_used / total_allocated * 100) if total_allocated > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 User Engagement Rate", f"{usage_rate:.1f}%")
            st.metric("🎯 Prediction Usage Rate", f"{usage_percentage:.1f}%")
        
        with col2:
            st.metric("🔢 Total Allocated Predictions", total_allocated)
            st.metric("✅ Total Used Predictions", total_used)
        
        # Tier-wise breakdown
        tier_stats = {}
        for user in users:
            tier = user['tier_display']
            if tier not in tier_stats:
                tier_stats[tier] = {
                    'users': 0,
                    'allocated': 0,
                    'used': 0,
                    'active_users': 0
                }
            
            tier_stats[tier]['users'] += 1
            tier_stats[tier]['allocated'] += user['max_predictions']
            tier_stats[tier]['used'] += user['predictions_used']
            if user['predictions_used'] > 0:
                tier_stats[tier]['active_users'] += 1
        
        st.subheader("🎭 Tier-wise Analytics")
        
        for tier, stats in tier_stats.items():
            with st.expander(f"📊 {tier} Analytics", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("👥 Users", stats['users'])
                with col2:
                    st.metric("🎯 Allocated", stats['allocated'])
                with col3:
                    st.metric("✅ Used", stats['used'])
                with col4:
                    engagement = (stats['active_users'] / stats['users'] * 100) if stats['users'] > 0 else 0
                    st.metric("📈 Engagement", f"{engagement:.1f}%")
    
    def individual_user_management(self):
        """Individual user management section"""
        st.header("👤 Individual User Management")
        st.markdown("---")
        
        # User lookup
        user_id_input = st.text_input("🔍 Enter User ID", placeholder="USER-XXXXXXXX")
        
        if user_id_input:
            user_details = self.user_manager.get_user_details(user_id_input)
            
            if user_details:
                st.success(f"✅ User found: {user_id_input}")
                
                # Display user details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 User Information")
                    st.info(f"**User ID:** {user_details['user_id']}")
                    st.info(f"**Tier:** {user_details['tier_display']}")
                    st.info(f"**Status:** {'✅ Active' if user_details['is_active'] else '❌ Inactive'}")
                    st.info(f"**Created:** {user_details['created_at'][:10] if user_details['created_at'] else 'N/A'}")
                    st.info(f"**Last Used:** {user_details['last_used'][:10] if user_details['last_used'] else 'Never'}")
                
                with col2:
                    st.subheader("🎯 Prediction Usage")
                    st.metric("Maximum Predictions", user_details['max_predictions'])
                    st.metric("Predictions Used", user_details['predictions_used'])
                    st.metric("Predictions Remaining", user_details['predictions_remaining'])
                    
                    if user_details['premium_key']:
                        st.code(f"Premium Key: {user_details['premium_key']}")
                
                # User management actions
                st.subheader("⚡ User Actions")
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("🔄 Reset Predictions", type="secondary"):
                        if self.user_manager.reset_user_predictions(user_id_input):
                            st.success("✅ Predictions reset successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset predictions")
                
                with action_col2:
                    if st.button("🎯 Use Prediction (Test)", type="secondary"):
                        if self.user_manager.use_prediction(user_id_input):
                            st.success("✅ Prediction used successfully!")
                            st.rerun()
                        else:
                            st.error("❌ No predictions remaining or user inactive")
                
                with action_col3:
                    new_status = "Deactivate" if user_details['is_active'] else "Activate"
                    if st.button(f"🔄 {new_status} User", type="secondary"):
                        # Toggle user status
                        with self.db_manager._get_connection() as conn:
                            conn.execute('''
                            UPDATE user_ids 
                            SET is_active = ?
                            WHERE user_id = ?
                            ''', (not user_details['is_active'], user_id_input))
                        
                        st.success(f"✅ User {new_status.lower()}d successfully!")
                        st.rerun()
                
                # Usage history
                st.subheader("📊 Usage History")
                with self.db_manager._get_connection() as conn:
                    cursor = conn.execute('''
                    SELECT action_type, timestamp, details
                    FROM usage_tracking 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                    ''', (user_id_input,))
                    
                    usage_history = cursor.fetchall()
                
                if usage_history:
                    history_data = []
                    for record in usage_history:
                        history_data.append({
                            'Action': record[0],
                            'Timestamp': record[1][:19] if record[1] else 'N/A',
                            'Details': record[2] or 'N/A'
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("📭 No usage history found for this user.")
            
            else:
                st.error("❌ User not found. Please check the User ID.")
    
    def admin_settings(self):
        """Admin settings and configuration"""
        st.header("⚙️ Admin Settings")
        st.markdown("---")
        
        # Password change
        st.subheader("🔐 Change Password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("🔄 Change Password", type="primary"):
                if not all([current_password, new_password, confirm_password]):
                    st.error("❌ All fields are required")
                elif new_password != confirm_password:
                    st.error("❌ New passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    # Validate current password
                    login_result = self.admin_manager.validate_login(
                        st.session_state.admin_username, 
                        current_password, 
                        '127.0.0.1'
                    )
                    
                    if login_result['valid']:
                        # Update password
                        hashed_password = self.admin_manager._hash_password(new_password)
                        
                        with self.db_manager._get_connection() as conn:
                            conn.execute('''
                            UPDATE admin_users 
                            SET password_hash = ?, salt = ?
                            WHERE username = ?
                            ''', (
                                hashed_password['password_hash'], 
                                hashed_password['salt'], 
                                st.session_state.admin_username
                            ))
                        
                        st.success("✅ Password updated successfully!")
                    else:
                        st.error("❌ Current password is incorrect")
        
        with col2:
            # System statistics
            st.subheader("📊 System Statistics")
            
            with self.db_manager._get_connection() as conn:
                # Total users
                cursor = conn.execute('SELECT COUNT(*) FROM user_ids')
                total_users = cursor.fetchone()[0]
                
                # Active users
                cursor = conn.execute('SELECT COUNT(*) FROM user_ids WHERE is_active = 1')
                active_users = cursor.fetchone()[0]
                
                # Total predictions used
                cursor = conn.execute('SELECT SUM(predictions_used) FROM user_ids')
                total_used = cursor.fetchone()[0] or 0
                
                # Premium keys generated
                cursor = conn.execute('SELECT COUNT(*) FROM premium_keys')
                premium_keys = cursor.fetchone()[0]
            
            st.metric("👥 Total Users", total_users)
            st.metric("✅ Active Users", active_users)
            st.metric("🎯 Total Predictions Used", total_used)
            st.metric("🔑 Premium Keys Generated", premium_keys)
    
    def logout(self):
        """Logout functionality"""
        # Reset session state
        st.session_state.admin_logged_in = False
        st.session_state.admin_username = None
        st.session_state.admin_role = None
        st.rerun()
    
    def main(self):
        """Main application flow"""
        # Set page configuration
        st.set_page_config(
            page_title="Admin Dashboard - User Management", 
            page_icon="🎯", 
            layout="wide"
        )
        
        # Check login status
        if not st.session_state.admin_logged_in:
            self.login_page()
            return
        
        # Dashboard header
        self.dashboard_header()
        
        # Main navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Generate Users", 
            "📊 User Management", 
            "👤 Individual User", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.user_generation_section()
        
        with tab2:
            self.user_management_section()
        
        with tab3:
            self.individual_user_management()
        
        with tab4:
            self.admin_settings()

# User-facing validation functions for the main app
def validate_premium_access(user_id: str, key: str) -> Dict[str, Any]:
    """
    Validate premium access for a user
    
    Args:
        user_id (str): User ID
        key (str): Premium key
    
    Returns:
        Dict[str, Any]: Validation result
    """
    db_manager = DatabaseManager()
    
    with db_manager._get_connection() as conn:
        # Check if user ID exists and is active
        cursor = conn.execute('''
        SELECT user_id, tier, max_predictions, predictions_used, 
               predictions_remaining, premium_key, is_active
        FROM user_ids 
        WHERE user_id = ? AND is_active = 1
        ''', (user_id,))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            return {
                'valid': False,
                'tier': 'free',
                'message': 'User ID not found or inactive'
            }
        
        # Extract user information
        (db_user_id, tier, max_predictions, predictions_used, 
         predictions_remaining, premium_key, is_active) = user_data
        
        # For free tier, no key required
        if tier == 'free':
            return {
                'valid': True,
                'tier': 'free',
                'user_id': user_id,
                'predictions_used': predictions_used,
                'max_predictions': max_predictions,
                'predictions_remaining': predictions_remaining,
                'message': 'Free tier access granted'
            }
        
        # For premium tiers, validate the key
        if premium_key and key == premium_key:
            return {
                'valid': True,
                'tier': 'premium',
                'user_id': user_id,
                'predictions_used': predictions_used,
                'max_predictions': max_predictions,
                'predictions_remaining': predictions_remaining,
                'next_reset': (datetime.now() + timedelta(days=30)).isoformat(),
                'message': 'Premium access granted successfully!'
            }
        
        return {
            'valid': False,
            'tier': 'free',
            'message': 'Invalid premium key for this user'
        }

def use_premium_prediction(user_id: str) -> bool:
    """
    Use a prediction for a user (decrement remaining predictions)
    
    Args:
        user_id (str): User ID
    
    Returns:
        bool: True if prediction was successfully used, False otherwise
    """
    db_manager = DatabaseManager()
    user_manager = EnhancedUserManager(db_manager)
    
    return user_manager.use_prediction(user_id)

def get_user_prediction_status(user_id: str) -> Dict[str, Any]:
    """
    Get user's prediction usage status
    
    Args:
        user_id (str): User ID
    
    Returns:
        Dict[str, Any]: User status information
    """
    db_manager = DatabaseManager()
    user_manager = EnhancedUserManager(db_manager)
    
    user_details = user_manager.get_user_details(user_id)
    
    if user_details:
        return {
            'user_id': user_details['user_id'],
            'tier': user_details['tier'],
            'predictions_used': user_details['predictions_used'],
            'max_predictions': user_details['max_predictions'],
            'predictions_remaining': user_details['predictions_remaining'],
            'next_reset': user_details['reset_date'],
            'is_active': user_details['is_active']
        }
    
    return {
        'user_id': user_id,
        'tier': 'unknown',
        'predictions_used': 0,
        'max_predictions': 0,
        'predictions_remaining': 0,
        'next_reset': 'Unknown',
        'is_active': False
    }

class UserIDManager:
    """
    Manages generation and tracking of user IDs
    """
    @staticmethod
    def generate_user_id() -> str:
        """
        Generate a unique user ID
        
        Returns:
            str: Unique user ID in format USER-XXXXXXXX
        """
        return f"USER-{uuid.uuid4().hex[:8].upper()}"
    
    @staticmethod
    def validate_user_id(user_id: str) -> Dict[str, Any]:
        """
        Validate user ID format and existence
        
        Args:
            user_id (str): User ID to validate
        
        Returns:
            Dict[str, Any]: Validation result
        """
        # Check format
        if not user_id or not user_id.startswith('USER-') or len(user_id) != 13:
            return {
                'valid': False,
                'message': 'Invalid User ID format. Expected: USER-XXXXXXXX'
            }
        
        # Check existence in database
        db_manager = DatabaseManager()
        with db_manager._get_connection() as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM user_ids WHERE user_id = ?', (user_id,))
            exists = cursor.fetchone()[0] > 0
        
        if exists:
            return {
                'valid': True,
                'message': 'User ID is valid and exists'
            }
        
        return {
            'valid': False,
            'message': 'User ID not found in system'
        }

class FeatureAccessControl:
    """
    Manage feature access based on user tiers
    """
    @staticmethod
    def get_feature_access(tier: str = 'free') -> Dict[str, Dict[str, Any]]:
        """
        Get feature access for different subscription tiers
        
        Args:
            tier (str): Subscription tier
        
        Returns:
            Dict[str, Dict[str, Any]]: Features available for the tier
        """
        feature_tiers = {
            'free': {
                'predictions': {
                    'models_available': 2,
                    'daily_predictions': 0,
                    'prediction_horizon': 'N/A',
                    'cross_validation': False,
                    'model_explanations': False
                },
                'analytics': {
                    'basic_charts': True,
                    'advanced_charting': False,
                    'technical_indicators': False,
                    'sentiment_analysis': False,
                    'alternative_data': False
                },
                'risk_management': {
                    'basic_risk_metrics': True,
                    'advanced_risk_analysis': False,
                    'portfolio_optimization': False,
                    'backtesting': False
                }
            },
            'tier_10': {
                'predictions': {
                    'models_available': 4,
                    'daily_predictions': 10,
                    'prediction_horizon': '3 Days',
                    'cross_validation': True,
                    'model_explanations': True
                },
                'analytics': {
                    'basic_charts': True,
                    'advanced_charting': True,
                    'technical_indicators': True,
                    'sentiment_analysis': False,
                    'alternative_data': False
                },
                'risk_management': {
                    'basic_risk_metrics': True,
                    'advanced_risk_analysis': True,
                    'portfolio_optimization': False,
                    'backtesting': False
                }
            },
            'tier_25': {
                'predictions': {
                    'models_available': 6,
                    'daily_predictions': 25,
                    'prediction_horizon': '7 Days',
                    'cross_validation': True,
                    'model_explanations': True
                },
                'analytics': {
                    'basic_charts': True,
                    'advanced_charting': True,
                    'technical_indicators': True,
                    'sentiment_analysis': True,
                    'alternative_data': False
                },
                'risk_management': {
                    'basic_risk_metrics': True,
                    'advanced_risk_analysis': True,
                    'portfolio_optimization': True,
                    'backtesting': True
                }
            },
            'tier_50': {
                'predictions': {
                    'models_available': 8,
                    'daily_predictions': 50,
                    'prediction_horizon': '14 Days',
                    'cross_validation': True,
                    'model_explanations': True
                },
                'analytics': {
                    'basic_charts': True,
                    'advanced_charting': True,
                    'technical_indicators': True,
                    'sentiment_analysis': True,
                    'alternative_data': True
                },
                'risk_management': {
                    'basic_risk_metrics': True,
                    'advanced_risk_analysis': True,
                    'portfolio_optimization': True,
                    'backtesting': True
                }
            },
            'tier_100': {
                'predictions': {
                    'models_available': 8,
                    'daily_predictions': 100,
                    'prediction_horizon': '30 Days',
                    'cross_validation': True,
                    'model_explanations': True
                },
                'analytics': {
                    'basic_charts': True,
                    'advanced_charting': True,
                    'technical_indicators': True,
                    'sentiment_analysis': True,
                    'alternative_data': True
                },
                'risk_management': {
                    'basic_risk_metrics': True,
                    'advanced_risk_analysis': True,
                    'portfolio_optimization': True,
                    'backtesting': True
                },
                'ai_capabilities': {
                    'drift_detection': True,
                    'regime_detection': True,
                    'meta_learning': True
                }
            }
        }
        
        # Map premium tiers to premium features
        if tier in ['tier_10', 'tier_25', 'tier_50', 'tier_100']:
            return feature_tiers.get(tier, feature_tiers['free'])
        
        # Default to free tier
        return feature_tiers.get(tier, feature_tiers['free'])

def main():
    """Run the admin dashboard application"""
    admin_app = AdminDashboardApp()
    admin_app.main()

if __name__ == "__main__":
    main()