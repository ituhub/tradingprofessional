#!/usr/bin/env python3
"""
Enhanced Admin Dashboard for User Management System
"""

import streamlit as st
# Set page configuration at the very beginning
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="🎯",
    layout="wide"
)

import logging
import pandas as pd
from datetime import datetime, timedelta
from premium_keys import (
    DatabaseManager,
    EnhancedUserManager as UserManager,
    SecureAdminManager as AdminManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('admin_dashboard.log')
    ]
)

logger = logging.getLogger(__name__)

class AdminDashboard:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.user_manager = UserManager(self.db_manager)
        self.admin_manager = AdminManager(self.db_manager)
        self._initialize_session_state()
        self._initialize_default_admin()
        self._setup_styling()

    def _initialize_session_state(self):
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
            st.session_state.admin_username = None
            st.session_state.admin_role = None

    def _initialize_default_admin(self):
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM admin_users')
                admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                self.admin_manager.create_admin_user(
                    username='admin',
                    password='admin123',
                    role='admin'
                )
                logger.info("Default admin created successfully")
        except Exception as e:
            logger.error(f"Error creating default admin: {e}")

    def _setup_styling(self):
        st.markdown("""
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .admin-header {
            padding: 2rem;
            background: linear-gradient(135deg, #0d6efd, #0dcaf0);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        if not st.session_state.admin_logged_in:
            self.render_login_page()
        else:
            self.render_dashboard()

    def render_login_page(self):
        st.markdown(
            '<div class="admin-header">'
            '<h1>🔐 Admin Dashboard</h1>'
            '<p>Secure Management System</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 👤 Administrator Login")
            
            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            if st.button("Login", type="primary"):
                login_result = self.admin_manager.validate_login(
                    username, password, '127.0.0.1'
                )
                
                if login_result['valid']:
                    st.session_state.admin_logged_in = True
                    st.session_state.admin_username = username
                    st.session_state.admin_role = login_result['role']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(login_result['message'])
            
            # Add help information
            with st.expander("ℹ️ Need Help?"):
                st.markdown("""
                **Default Credentials:**
                - Username: `admin`
                - Password: `admin123`
                
                **Security Notice:**
                - Change default password after first login
                - Maximum 5 login attempts allowed
                - Account locks for 15 minutes after failed attempts
                """)

    def render_dashboard(self):
        st.markdown(
            '<div class="admin-header">'
            '<h1>🎯 Admin Dashboard</h1>'
            f'<p>Welcome, {st.session_state.admin_username}</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.sidebar.button("Logout"):
            st.session_state.admin_logged_in = False
            st.session_state.admin_username = None
            st.session_state.admin_role = None
            st.rerun()

        tab1, tab2, tab3 = st.tabs(["Generate Users", "User Management", "Settings"])

        with tab1:
            self.render_user_generation_tab()
        
        with tab2:
            self.render_user_management_tab()
        
        with tab3:
            self.render_settings_tab()

    def render_user_generation_tab(self):
        st.header("User Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Premium Tier")
            tier = st.selectbox(
                "Select Premium Tier",
                options=['tier_10', 'tier_25', 'tier_50', 'tier_100'],
                format_func=lambda x: self.user_manager.tier_configs[x]['display_name']
            )
            count = st.number_input("Number of Users", min_value=1, max_value=50, value=1)
            
            if st.button("Generate Premium Users"):
                users = self.user_manager.generate_user_id_with_tier(tier, count)
                st.success(f"Generated {len(users)} users!")
                
                # Create DataFrame for download
                df = pd.DataFrame(users)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"premium_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
                for user in users:
                    with st.expander(f"User: {user['user_id']}", expanded=True):
                        st.code(f"User ID: {user['user_id']}")
                        st.code(f"Premium Key: {user['premium_key']}")
                        st.code(f"Tier: {user['tier_display']}")

        with col2:
            st.subheader("Free Tier")
            free_count = st.number_input(
                "Number of Free Users",
                min_value=1,
                max_value=50,
                value=1,
                key="free_count"
            )
            
            if st.button("Generate Free Users"):
                users = self.user_manager.generate_user_id_with_tier('free', free_count)
                st.success(f"Generated {len(users)} users!")
                
                # Create DataFrame for download
                df = pd.DataFrame(users)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"free_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
                for user in users:
                    with st.expander(f"User: {user['user_id']}", expanded=True):
                        st.code(f"User ID: {user['user_id']}")
                        st.code(f"Tier: {user['tier_display']}")

    def render_user_management_tab(self):
        st.header("User Management")
        
        # Search and filters
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Search User ID")
        with col2:
            status_filter = st.selectbox(
                "Filter by Status",
                options=['All', 'Active', 'Inactive']
            )
        
        users = self.user_manager.get_all_users()
        
        # Apply filters
        if search:
            users = [u for u in users if search.upper() in u['user_id'].upper()]
        if status_filter != 'All':
            is_active = status_filter == 'Active'
            users = [u for u in users if u['is_active'] == is_active]
        
        if users:
            # Display statistics
            total_users = len(users)
            active_users = len([u for u in users if u['is_active']])
            total_predictions = sum(u['predictions_used'] for u in users)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", total_users)
            with col2:
                st.metric("Active Users", active_users)
            with col3:
                st.metric("Total Predictions Used", total_predictions)
            
            # User table
            df = pd.DataFrame([{
                'User ID': u['user_id'],
                'Tier': u['tier_display'],
                'Used': u['predictions_used'],
                'Remaining': u['predictions_remaining'],
                'Status': "✅ Active" if u['is_active'] else "❌ Inactive"
            } for u in users])
            
            st.dataframe(df, use_container_width=True)
            
            # User actions
            selected_user = st.selectbox("Select User for Actions", df['User ID'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset Predictions"):
                    if self.user_manager.reset_user_predictions(selected_user):
                        st.success("✅ Predictions reset successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to reset predictions")
            
            # Export option
            with col2:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Export to CSV",
                    csv,
                    f"users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        else:
            st.info("No users found")

    def render_settings_tab(self):
        st.header("Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Change Password"):
                if new_password != confirm_password:
                    st.error("❌ New passwords do not match!")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    login_result = self.admin_manager.validate_login(
                        st.session_state.admin_username,
                        current_password,
                        '127.0.0.1'
                    )
                    
                    if login_result['valid']:
                        try:
                            hashed = self.admin_manager._hash_password(new_password)
                            with self.db_manager._get_connection() as conn:
                                conn.execute('''
                                UPDATE admin_users 
                                SET password_hash = ?, salt = ?
                                WHERE username = ?
                                ''', (
                                    hashed['password_hash'],
                                    hashed['salt'],
                                    st.session_state.admin_username
                                ))
                            st.success("✅ Password updated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error updating password: {e}")
                    else:
                        st.error("❌ Current password is incorrect!")
        
        with col2:
            st.subheader("System Statistics")
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM user_ids')
                total_users = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM user_ids WHERE is_active = 1')
                active_users = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT SUM(predictions_used) FROM user_ids')
                total_predictions = cursor.fetchone()[0] or 0
            
            st.metric("👥 Total Users", total_users)
            st.metric("✅ Active Users", active_users)
            st.metric("🎯 Total Predictions Used", total_predictions)

def main():
    dashboard = AdminDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()