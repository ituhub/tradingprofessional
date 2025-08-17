#!/usr/bin/env python3
"""
Synchronized Admin Dashboard for User Management System
Compatible with user application and provides comprehensive user management
"""

import os
import sys
import logging
import pandas as pd
import sqlite3
import streamlit as st
import hashlib
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import the synchronized UserDatabase
from user_database import UserDatabase, create_default_admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('admin_dashboard.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class AdminAuthenticator:
    """
    Secure admin authentication system
    """
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
        self.logger = logging.getLogger('AdminAuth')
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = uuid.uuid4().hex
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return {'salt': salt, 'password_hash': password_hash}
    
    def validate_admin_login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Validate admin login credentials
        
        Args:
            username (str): Admin username
            password (str): Admin password
        
        Returns:
            Dict[str, Any]: Validation result
        """
        try:
            with sqlite3.connect(self.user_db.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT username, password_hash, salt, role, is_active, is_locked, 
                       failed_attempts, lockout_time
                FROM admin_users
                WHERE username = ?
                ''', (username,))
                
                admin_data = cursor.fetchone()
                
                if not admin_data:
                    self.logger.warning(f"Login attempt for non-existent admin: {username}")
                    return {'valid': False, 'message': 'Invalid credentials'}
                
                (db_username, db_password_hash, db_salt, role, 
                 is_active, is_locked, failed_attempts, lockout_time) = admin_data
                
                # Check account status
                if not is_active:
                    return {'valid': False, 'message': 'Account is inactive'}
                
                # Check lockout
                if is_locked and lockout_time:
                    try:
                        lockout_dt = datetime.fromisoformat(lockout_time)
                        if lockout_dt > datetime.now():
                            return {'valid': False, 'message': 'Account is locked. Try again later.'}
                    except ValueError:
                        pass
                
                # Verify password
                hashed_input = self._hash_password(password, db_salt)
                
                if hashed_input['password_hash'] != db_password_hash:
                    # Increment failed attempts
                    new_failed_attempts = (failed_attempts or 0) + 1
                    
                    if new_failed_attempts >= 5:
                        # Lock account for 15 minutes
                        lockout_time = datetime.now() + timedelta(minutes=15)
                        cursor.execute('''
                        UPDATE admin_users 
                        SET failed_attempts = ?, is_locked = 1, lockout_time = ?
                        WHERE username = ?
                        ''', (new_failed_attempts, lockout_time.isoformat(), username))
                        
                        conn.commit()
                        return {'valid': False, 'message': 'Too many failed attempts. Account locked.'}
                    else:
                        cursor.execute('''
                        UPDATE admin_users 
                        SET failed_attempts = ?
                        WHERE username = ?
                        ''', (new_failed_attempts, username))
                        conn.commit()
                    
                    return {'valid': False, 'message': 'Invalid credentials'}
                
                # Successful login - reset failed attempts
                cursor.execute('''
                UPDATE admin_users 
                SET failed_attempts = 0, is_locked = 0, lockout_time = NULL, last_login = ?
                WHERE username = ?
                ''', (datetime.now().isoformat(), username))
                
                conn.commit()
                
                self.logger.info(f"Successful admin login: {username}")
                
                return {
                    'valid': True,
                    'username': username,
                    'role': role,
                    'message': 'Login successful'
                }
        
        except Exception as e:
            self.logger.error(f"Error validating admin login: {e}")
            return {'valid': False, 'message': 'Authentication error'}
    
    def create_admin_user(self, username: str, password: str, role: str = 'admin') -> bool:
        """
        Create new admin user
        
        Args:
            username (str): Admin username
            password (str): Admin password
            role (str): Admin role
        
        Returns:
            bool: True if created successfully
        """
        try:
            with sqlite3.connect(self.user_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if username exists
                cursor.execute('SELECT COUNT(*) FROM admin_users WHERE username = ?', (username,))
                if cursor.fetchone()[0] > 0:
                    return False
                
                # Hash password
                hashed_data = self._hash_password(password)
                
                # Create admin user
                cursor.execute('''
                INSERT INTO admin_users 
                (username, password_hash, salt, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ''', (
                    username,
                    hashed_data['password_hash'],
                    hashed_data['salt'],
                    role,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.info(f"Admin user created: {username}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error creating admin user: {e}")
            return False

class AdminDashboard:
    """
    Main admin dashboard application with full synchronization
    """
    
    def __init__(self, db_path: str = 'user_management.db'):
        """
        Initialize admin dashboard
        
        Args:
            db_path (str): Database path
        """
        try:
            # Initialize database
            self.user_db = UserDatabase(db_path)
            self.authenticator = AdminAuthenticator(self.user_db)
            
            # Initialize session state
            self._initialize_session_state()
            
            # Ensure default admin exists
            self._ensure_default_admin()
            
            # Configure tier details for UI
            self.tier_configs = {
                'free': {
                    'display_name': 'Free Tier',
                    'max_predictions': 0,
                    'color': '#6c757d'
                },
                'tier_10': {
                    'display_name': '10 Predictions Tier',
                    'max_predictions': 10,
                    'color': '#17a2b8'
                },
                'tier_25': {
                    'display_name': '25 Predictions Tier',
                    'max_predictions': 25,
                    'color': '#28a745'
                },
                'tier_50': {
                    'display_name': '50 Predictions Tier',
                    'max_predictions': 50,
                    'color': '#ffc107'
                },
                'tier_100': {
                    'display_name': '100 Predictions Tier',
                    'max_predictions': 100,
                    'color': '#dc3545'
                }
            }
            
            logger.info("Admin Dashboard initialized successfully")
        
        except Exception as e:
            logger.error(f"Dashboard initialization error: {e}")
            st.error(f"Failed to initialize admin dashboard: {e}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'admin_logged_in': False,
            'admin_username': None,
            'admin_role': None,
            'admin_login_time': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _ensure_default_admin(self):
        """Ensure default admin user exists"""
        try:
            create_default_admin(self.user_db.db_path)
        except Exception as e:
            logger.error(f"Error ensuring default admin: {e}")
    
    def run(self):
        """Main dashboard execution"""
        # Page configuration
        st.set_page_config(
            page_title="Admin Dashboard - User Management",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self._apply_custom_css()
        
        # Check authentication
        if not st.session_state.admin_logged_in:
            self._render_login_page()
        else:
            self._render_dashboard()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
            margin: 1rem 0;
        }
        .tier-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            color: white;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .info-box {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_login_page(self):
        """Render admin login page"""
        st.markdown("""
        <div class="main-header">
            <h1>🔐 Admin Dashboard Login</h1>
            <p>User Management System Administration</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔑 Administrator Authentication")
            
            with st.form("admin_login_form"):
                username = st.text_input("👤 Username", placeholder="Enter admin username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
                login_button = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
                
                if login_button:
                    if username and password:
                        result = self.authenticator.validate_admin_login(username, password)
                        
                        if result['valid']:
                            st.session_state.admin_logged_in = True
                            st.session_state.admin_username = username
                            st.session_state.admin_role = result['role']
                            st.session_state.admin_login_time = datetime.now()
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                    else:
                        st.error("❌ Please enter both username and password")
        
        # Information section
        with st.expander("ℹ️ Default Credentials & Information", expanded=False):
            st.markdown("""
            **🔧 Default Admin Credentials:**
            - Username: `admin`
            - Password: `admin123`
            
            **🛡️ Security Features:**
            - Account lockout after 5 failed attempts
            - 15-minute lockout duration
            - Secure password hashing with PBKDF2
            - Session management and logging
            
            **⚠️ Important:** Change the default password after first login!
            """)
    
    def _render_dashboard(self):
        """Render main dashboard interface"""
        # Dashboard header
        self._render_dashboard_header()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "👥 Generate Users",
            "📊 User Management", 
            "👤 Individual User",
            "📈 Analytics",
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_user_generation_tab()
        
        with tab2:
            self._render_user_management_tab()
        
        with tab3:
            self._render_individual_user_tab()
        
        with tab4:
            self._render_analytics_tab()
        
        with tab5:
            self._render_settings_tab()
    
    def _render_dashboard_header(self):
        """Render dashboard header with admin info"""
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown("""
            <div class="main-header">
                <h2>🎯 Admin Dashboard</h2>
                <p>User Management System</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            login_time = st.session_state.admin_login_time
            login_str = login_time.strftime('%Y-%m-%d %H:%M') if login_time else 'Unknown'
            
            st.markdown(f"""
            <div class="info-box">
                <strong>👤 Admin:</strong> {st.session_state.admin_username}<br>
                <strong>🏷️ Role:</strong> {st.session_state.admin_role}<br>
                <strong>🕐 Login:</strong> {login_str}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                self._logout()
        
        st.markdown("---")
    
    def _render_user_generation_tab(self):
        """Render user generation interface"""
        st.header("👥 User ID & Premium Key Generation")
        
        col1, col2 = st.columns(2)
        
        # Premium Tier Generation
        with col1:
            st.subheader("🎖️ Premium Tier Generation")
            
            premium_tiers = ['tier_10', 'tier_25', 'tier_50', 'tier_100']
            selected_tier = st.selectbox(
                "Select Premium Tier",
                options=premium_tiers,
                format_func=lambda x: self.tier_configs[x]['display_name']
            )
            
            premium_count = st.number_input(
                "Number of Premium Users",
                min_value=1,
                max_value=50,
                value=1
            )
            
            if st.button("🎖️ Generate Premium Users", type="primary"):
                self._generate_users(selected_tier, premium_count)
        
        # Free Tier Generation
        with col2:
            st.subheader("🆓 Free Tier Generation")
            
            free_count = st.number_input(
                "Number of Free Users",
                min_value=1,
                max_value=50,
                value=1,
                key="free_count"
            )
            
            st.info("📌 Free tier users get 0 predictions but are tracked for analytics.")
            
            if st.button("🆓 Generate Free Users", type="secondary"):
                self._generate_users('free', free_count)
    
    def _generate_users(self, tier: str, count: int):
        """Generate users with specified tier"""
        try:
            generated_users = []
            
            for i in range(count):
                # Generate user ID
                user_id = f"USER-{uuid.uuid4().hex[:8].upper()}"
                
                # Generate premium key for non-free tiers
                premium_key = None
                if tier != 'free':
                    premium_key = self.user_db.create_premium_key(
                        tier, 
                        created_by=st.session_state.admin_username
                    )
                
                # Create user
                success = self.user_db.create_user(
                    user_id=user_id,
                    tier=tier,
                    premium_key=premium_key,
                    created_by=st.session_state.admin_username
                )
                
                if success:
                    tier_config = self.tier_configs[tier]
                    generated_users.append({
                        'user_id': user_id,
                        'tier': tier,
                        'tier_display': tier_config['display_name'],
                        'max_predictions': tier_config['max_predictions'],
                        'premium_key': premium_key,
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                    })
            
            if generated_users:
                st.success(f"✅ Generated {len(generated_users)} users successfully!")
                
                # Display generated users
                for user in generated_users:
                    tier_color = self.tier_configs[user['tier']]['color']
                    
                    with st.expander(f"📋 {user['user_id']} - {user['tier_display']}", expanded=True):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.code(f"User ID: {user['user_id']}")
                            st.markdown(f"""
                            <span class="tier-badge" style="background-color: {tier_color}">
                                {user['tier_display']}
                            </span>
                            """, unsafe_allow_html=True)
                            st.code(f"Max Predictions: {user['max_predictions']}")
                        
                        with col_b:
                            if user['premium_key']:
                                st.code(f"Premium Key: {user['premium_key']}")
                            st.code(f"Created: {user['created_at']}")
                
                # Download option
                if st.checkbox("💾 Download as CSV"):
                    df = pd.DataFrame(generated_users)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Users CSV",
                        data=csv,
                        file_name=f"{tier}_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Failed to generate users")
        
        except Exception as e:
            st.error(f"❌ Error generating users: {e}")
            logger.error(f"User generation error: {e}")
    
    def _render_user_management_tab(self):
        """Render user management interface"""
        st.header("📊 User Management & Tracking")
        
        # Get all users and statistics
        all_users = self.user_db.get_all_users()
        stats = self.user_db.get_usage_statistics()
        
        # Statistics overview
        st.subheader("📈 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Users</h3>
                <h2>{stats.get('total_users', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Active Users</h3>
                <h2>{stats.get('active_users', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Predictions Used</h3>
                <h2>{stats.get('total_predictions_used', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            usage_rate = stats.get('usage_rate', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Usage Rate</h3>
                <h2>{usage_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Tier breakdown
        tier_breakdown = stats.get('tier_breakdown', [])
        if tier_breakdown:
            st.subheader("🎭 Tier Distribution")
            
            tier_cols = st.columns(len(tier_breakdown))
            for i, tier_data in enumerate(tier_breakdown):
                with tier_cols[i]:
                    tier_color = self.tier_configs.get(tier_data['tier'], {}).get('color', '#6c757d')
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="tier-badge" style="background-color: {tier_color}">
                            {tier_data['display_name'] or tier_data['tier']}
                        </span>
                        <h3>{tier_data['user_count']} Users</h3>
                        <p>Used: {tier_data['predictions_used']}/{tier_data['predictions_allocated']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # User search and filtering
        st.subheader("🔍 Search & Filter Users")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔎 Search User ID", placeholder="USER-XXXXXXXX")
        
        with col2:
            tier_filter = st.selectbox(
                "🎭 Filter by Tier",
                options=["All"] + list(self.tier_configs.keys()),
                format_func=lambda x: "All Tiers" if x == "All" else self.tier_configs[x]['display_name']
            )
        
        with col3:
            status_filter = st.selectbox(
                "🔄 Filter by Status",
                options=["All", "Active", "Inactive"]
            )
        
        # Apply filters
        filtered_users = all_users.copy()
        
        if search_term:
            filtered_users = [u for u in filtered_users if search_term.upper() in u['user_id'].upper()]
        
        if tier_filter != "All":
            filtered_users = [u for u in filtered_users if u['tier'] == tier_filter]
        
        if status_filter != "All":
            is_active = status_filter == "Active"
            filtered_users = [u for u in filtered_users if u['is_active'] == is_active]
        
        # Display filtered users
        st.subheader(f"👥 User List ({len(filtered_users)} users)")
        
        if filtered_users:
            # Create display data
            display_data = []
            for user in filtered_users:
                tier_config = self.tier_configs.get(user['tier'], {})
                display_data.append({
                    'User ID': user['user_id'],
                    'Tier': tier_config.get('display_name', user['tier']),
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
                        if self.user_db.reset_user_predictions(
                            user['user_id'], 
                            st.session_state.admin_username
                        ):
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
                    self._generate_usage_report(filtered_users)
        
        else:
            st.info("🔭 No users match the current filters.")
    
    def _generate_usage_report(self, users: List[Dict]):
        """Generate detailed usage report"""
        st.subheader("📈 Detailed Usage Report")
        
        # Calculate metrics
        total_users = len(users)
        active_users = len([u for u in users if u['predictions_used'] > 0])
        engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
        
        total_allocated = sum(u['max_predictions'] for u in users)
        total_used = sum(u['predictions_used'] for u in users)
        usage_percentage = (total_used / total_allocated * 100) if total_allocated > 0 else 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 User Engagement Rate</h4>
                <h2>{engagement_rate:.1f}%</h2>
                <p>{active_users} out of {total_users} users have made predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Prediction Usage Rate</h4>
                <h2>{usage_percentage:.1f}%</h2>
                <p>{total_used} out of {total_allocated} predictions used</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tier-wise breakdown
        tier_stats = {}
        for user in users:
            tier = user['tier']
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
            tier_config = self.tier_configs.get(tier, {})
            tier_name = tier_config.get('display_name', tier)
            tier_color = tier_config.get('color', '#6c757d')
            
            with st.expander(f"📊 {tier_name} Analytics", expanded=True):
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
    
    def _render_individual_user_tab(self):
        """Render individual user management interface"""
        st.header("👤 Individual User Management")
        
        # User lookup
        user_id_input = st.text_input("🔍 Enter User ID", placeholder="USER-XXXXXXXX")
        
        if user_id_input:
            user = self.user_db.get_user(user_id_input)
            
            if user:
                st.success(f"✅ User found: {user_id_input}")
                
                # Display user details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 User Information")
                    tier_config = self.tier_configs.get(user['tier'], {})
                    tier_color = tier_config.get('color', '#6c757d')
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>🆔 User ID:</strong> {user['user_id']}<br>
                        <strong>🎭 Tier:</strong> <span class="tier-badge" style="background-color: {tier_color}">{tier_config.get('display_name', user['tier'])}</span><br>
                        <strong>🔄 Status:</strong> {'✅ Active' if user['is_active'] else '❌ Inactive'}<br>
                        <strong>📅 Created:</strong> {user['created_at'][:10] if user['created_at'] else 'N/A'}<br>
                        <strong>🕐 Last Used:</strong> {user['last_used'][:10] if user['last_used'] else 'Never'}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("🎯 Prediction Usage")
                    
                    # Create progress bar for prediction usage
                    usage_percent = (user['predictions_used'] / user['max_predictions'] * 100) if user['max_predictions'] > 0 else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Prediction Usage</h4>
                        <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; margin: 10px 0;">
                            <div style="background-color: #007bff; height: 100%; width: {usage_percent}%; border-radius: 10px;"></div>
                        </div>
                        <p><strong>Used:</strong> {user['predictions_used']} / {user['max_predictions']} ({usage_percent:.1f}%)</p>
                        <p><strong>Remaining:</strong> {user['predictions_remaining']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if user['premium_key']:
                        st.code(f"Premium Key: {user['premium_key']}")
                
                # User management actions
                st.subheader("⚡ User Actions")
                
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                
                with action_col1:
                    if st.button("🔄 Reset Predictions", type="secondary"):
                        if self.user_db.reset_user_predictions(user_id_input, st.session_state.admin_username):
                            st.success("✅ Predictions reset successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset predictions")
                
                with action_col2:
                    if st.button("🎯 Test Prediction Use", type="secondary"):
                        if self.user_db.use_prediction(user_id_input):
                            st.success("✅ Prediction used successfully!")
                            st.rerun()
                        else:
                            st.error("❌ No predictions remaining or user inactive")
                
                with action_col3:
                    new_status_text = "Deactivate" if user['is_active'] else "Activate"
                    if st.button(f"🔄 {new_status_text} User", type="secondary"):
                        try:
                            with sqlite3.connect(self.user_db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                UPDATE users 
                                SET is_active = ?, updated_at = ?
                                WHERE user_id = ?
                                ''', (not user['is_active'], datetime.now().isoformat(), user_id_input))
                                conn.commit()
                            
                            # Log action
                            self.user_db._log_action(
                                user_id_input,
                                'status_changed',
                                f'User {new_status_text.lower()}d by admin',
                                st.session_state.admin_username
                            )
                            
                            st.success(f"✅ User {new_status_text.lower()}d successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error updating user status: {e}")
                
                with action_col4:
                    # Tier change dropdown
                    new_tier = st.selectbox(
                        "Change Tier",
                        options=list(self.tier_configs.keys()),
                        index=list(self.tier_configs.keys()).index(user['tier']),
                        format_func=lambda x: self.tier_configs[x]['display_name'],
                        key="tier_change"
                    )
                    
                    if st.button("🔄 Update Tier", type="secondary"):
                        if new_tier != user['tier']:
                            if self.user_db.update_user_tier(user_id_input, new_tier, st.session_state.admin_username):
                                st.success(f"✅ Tier updated to {self.tier_configs[new_tier]['display_name']}!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update tier")
                
                # Usage history
                st.subheader("📊 Usage History")
                usage_history = self.user_db.get_user_usage_history(user_id_input, limit=20)
                
                if usage_history:
                    history_data = []
                    for record in usage_history:
                        history_data.append({
                            'Action': record['action_type'].replace('_', ' ').title(),
                            'Timestamp': record['timestamp'][:19] if record['timestamp'] else 'N/A',
                            'Details': record['details'] or 'N/A'
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("🔭 No usage history found for this user.")
            
            else:
                st.error("❌ User not found. Please check the User ID.")
    
    def _render_analytics_tab(self):
        """Render analytics and reporting interface"""
        st.header("📈 Analytics & Reporting")
        
        # Get comprehensive statistics
        stats = self.user_db.get_usage_statistics()
        
        # Time-based analytics
        st.subheader("📊 System Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Total Users</h4>
                <h1>{stats.get('total_users', 0)}</h1>
                <p>Registered in system</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Total Predictions</h4>
                <h1>{stats.get('total_predictions_used', 0)}</h1>
                <p>Predictions consumed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            usage_rate = stats.get('usage_rate', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 System Utilization</h4>
                <h1>{usage_rate:.1f}%</h1>
                <p>Overall usage rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        recent_activity = stats.get('recent_activity', [])
        if recent_activity:
            st.subheader("🔄 Recent Activity (Last 7 Days)")
            
            activity_data = []
            for activity in recent_activity:
                activity_data.append({
                    'Action Type': activity['action'].replace('_', ' ').title(),
                    'Count': activity['count']
                })
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                st.dataframe(activity_df, use_container_width=True)
        
        # Database health
        st.subheader("🔧 Database Health")
        
        try:
            with sqlite3.connect(self.user_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                db_size_mb = db_size / (1024 * 1024)
                
                # Table counts
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM usage_tracking")
                tracking_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM premium_keys")
                key_count = cursor.fetchone()[0]
                
                health_col1, health_col2, health_col3, health_col4 = st.columns(4)
                
                with health_col1:
                    st.metric("💾 Database Size", f"{db_size_mb:.2f} MB")
                
                with health_col2:
                    st.metric("👥 User Records", user_count)
                
                with health_col3:
                    st.metric("📊 Tracking Records", tracking_count)
                
                with health_col4:
                    st.metric("🔑 Premium Keys", key_count)
        
        except Exception as e:
            st.error(f"❌ Error retrieving database health: {e}")
        
        # Export options
        st.subheader("📥 Data Export")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Export Full User Report", type="secondary"):
                all_users = self.user_db.get_all_users()
                if all_users:
                    # Prepare export data
                    export_data = []
                    for user in all_users:
                        tier_config = self.tier_configs.get(user['tier'], {})
                        export_data.append({
                            'User ID': user['user_id'],
                            'Tier': tier_config.get('display_name', user['tier']),
                            'Max Predictions': user['max_predictions'],
                            'Used Predictions': user['predictions_used'],
                            'Remaining Predictions': user['predictions_remaining'],
                            'Premium Key': user['premium_key'] or 'N/A',
                            'Created At': user['created_at'],
                            'Last Used': user['last_used'] or 'Never',
                            'Is Active': 'Yes' if user['is_active'] else 'No'
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download User Report",
                        data=csv,
                        file_name=f"full_user_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No users found to export")
        
        with export_col2:
            if st.button("💾 Backup Database", type="secondary"):
                try:
                    backup_path = self.user_db.backup_database()
                    
                    with open(backup_path, 'rb') as f:
                        backup_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Database Backup",
                        data=backup_data,
                        file_name=f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                        mime="application/octet-stream"
                    )
                    
                    st.success(f"✅ Database backup created: {backup_path}")
                except Exception as e:
                    st.error(f"❌ Backup failed: {e}")
    
    def _render_settings_tab(self):
        """Render settings and configuration interface"""
        st.header("⚙️ Settings & Configuration")
        
        # Admin account management
        st.subheader("👤 Admin Account Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔒 Change Password")
            
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("🔄 Change Password", type="primary"):
                    if not all([current_password, new_password, confirm_password]):
                        st.error("❌ All fields are required")
                    elif new_password != confirm_password:
                        st.error("❌ New passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        # Validate current password
                        validation = self.authenticator.validate_admin_login(
                            st.session_state.admin_username,
                            current_password
                        )
                        
                        if validation['valid']:
                            # Update password
                            try:
                                hashed_data = self.authenticator._hash_password(new_password)
                                
                                with sqlite3.connect(self.user_db.db_path) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                    UPDATE admin_users 
                                    SET password_hash = ?, salt = ?
                                    WHERE username = ?
                                    ''', (
                                        hashed_data['password_hash'],
                                        hashed_data['salt'],
                                        st.session_state.admin_username
                                    ))
                                    conn.commit()
                                
                                st.success("✅ Password updated successfully!")
                            except Exception as e:
                                st.error(f"❌ Error updating password: {e}")
                        else:
                            st.error("❌ Current password is incorrect")
        
        with col2:
            st.markdown("#### 👥 Create New Admin")
            
            with st.form("create_admin_form"):
                new_admin_username = st.text_input("New Admin Username")
                new_admin_password = st.text_input("New Admin Password", type="password")
                new_admin_role = st.selectbox("Admin Role", ["admin", "moderator"])
                
                if st.form_submit_button("➕ Create Admin", type="secondary"):
                    if not all([new_admin_username, new_admin_password]):
                        st.error("❌ All fields are required")
                    elif len(new_admin_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        if self.authenticator.create_admin_user(new_admin_username, new_admin_password, new_admin_role):
                            st.success(f"✅ Admin user '{new_admin_username}' created successfully!")
                        else:
                            st.error("❌ Failed to create admin user (username may already exist)")
        
        # System configuration
        st.subheader("🔧 System Configuration")
        
        # Tier configuration display
        st.markdown("#### 🎭 Tier Configurations")
        
        tier_configs = self.user_db.get_tier_configs()
        if tier_configs:
            config_data = []
            for tier_config in tier_configs:
                config_data.append({
                    'Tier': tier_config['tier'],
                    'Display Name': tier_config['display_name'],
                    'Max Predictions': tier_config['max_predictions'],
                    'Price': f"${tier_config.get('price', 0):.2f}",
                    'Duration (Days)': tier_config.get('duration_days', 30),
                    'Features': len(tier_config.get('features', {}))
                })
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True)
        
        # Database maintenance
        st.subheader("🔧 Database Maintenance")
        
        maint_col1, maint_col2, maint_col3 = st.columns(3)
        
        with maint_col1:
            if st.button("🧹 Cleanup Expired Sessions", type="secondary"):
                try:
                    self.user_db.cleanup_expired_sessions()
                    st.success("✅ Expired sessions cleaned up")
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {e}")
        
        with maint_col2:
            if st.button("📊 Refresh Statistics", type="secondary"):
                try:
                    # Force refresh by clearing any cached data
                    st.cache_data.clear()
                    st.success("✅ Statistics refreshed")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Refresh failed: {e}")
        
        with maint_col3:
            if st.button("🔄 Reset Demo Data", type="secondary"):
                if st.checkbox("⚠️ Confirm reset (this will delete all demo users)"):
                    try:
                        # This would implement demo data reset logic
                        st.warning("⚠️ Demo data reset not implemented for safety")
                    except Exception as e:
                        st.error(f"❌ Reset failed: {e}")
        
        # Application info
        st.subheader("ℹ️ Application Information")
        
        info_data = {
            'Database Path': self.user_db.db_path,
            'Database Size': f"{os.path.getsize(self.user_db.db_path) / (1024*1024):.2f} MB" if os.path.exists(self.user_db.db_path) else "N/A",
            'Python Version': sys.version.split()[0],
            'Streamlit Version': st.__version__,
            'Admin Session': st.session_state.admin_username,
            'Login Time': st.session_state.admin_login_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.admin_login_time else 'N/A'
        }
        
        for key, value in info_data.items():
            st.text(f"{key}: {value}")
    
    def _logout(self):
        """Handle admin logout"""
        # Log logout action
        logger.info(f"Admin logout: {st.session_state.admin_username}")
        
        # Clear session state
        st.session_state.admin_logged_in = False
        st.session_state.admin_username = None
        st.session_state.admin_role = None
        st.session_state.admin_login_time = None
        
        st.success("✅ Logged out successfully!")
        st.rerun()

def main():
    """Main application entry point"""
    try:
        # Initialize and run dashboard
        dashboard = AdminDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Critical error: {e}")
        logger.critical(f"Critical application error: {e}")

if __name__ == "__main__":
    main()
