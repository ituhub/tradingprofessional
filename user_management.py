# user_management.py - Add this as a new file to your project

import streamlit as st
import pandas as pd
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid
import hashlib
import os
from pathlib import Path

class UserManager:
    """Enhanced User Management System for AI Trading Platform"""
    
    def __init__(self, data_file="users_data.json"):
        self.data_file = data_file
        self.users = self.load_users()
        
    def load_users(self) -> Dict:
        """Load users from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading users: {e}")
        
        return self.create_initial_users()
    
    def save_users(self):
        """Save users to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.users, f, default=str, indent=2)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def create_initial_users(self) -> Dict:
        """Create initial set of users"""
        users = {}
        for i in range(15):
            user_id = f"USER_{str(i + 1).zfill(3)}"
            users[user_id] = {
                'id': user_id,
                'name': f'User {i + 1}',
                'email': f'user{i + 1}@example.com',
                'usage': 0,
                'monthly_limit': 10,
                'status': 'active',
                'tier': 'premium' if i < 3 else 'free',  # First 3 are premium
                'created': datetime.now().isoformat(),
                'last_used': None,
                'usage_history': [],
                'api_key': self.generate_api_key()
            }
        return users
    
    def generate_api_key(self) -> str:
        """Generate unique API key for user"""
        return f"atps_{uuid.uuid4().hex[:16]}"  # AI Trading Platform Streamlit
    
    def add_user(self, name: str = None, email: str = None) -> str:
        """Add new user"""
        user_count = len(self.users)
        user_id = f"USER_{str(user_count + 1).zfill(3)}"
        
        self.users[user_id] = {
            'id': user_id,
            'name': name or f'User {user_count + 1}',
            'email': email or f'user{user_count + 1}@example.com',
            'usage': 0,
            'monthly_limit': 10,
            'status': 'active',
            'tier': 'free',
            'created': datetime.now().isoformat(),
            'last_used': None,
            'usage_history': [],
            'api_key': self.generate_api_key()
        }
        
        self.save_users()
        return user_id
    
    def validate_user(self, user_id: str) -> Dict[str, Any]:
        """Validate user access and return status"""
        if user_id not in self.users:
            return {'valid': False, 'reason': 'User not found'}
        
        user = self.users[user_id]
        
        if user['status'] != 'active':
            return {'valid': False, 'reason': 'User account suspended'}
        
        if user['usage'] >= user['monthly_limit']:
            return {'valid': False, 'reason': 'Monthly usage limit exceeded'}
        
        return {
            'valid': True, 
            'user': user,
            'remaining_usage': user['monthly_limit'] - user['usage']
        }
    
    def record_usage(self, user_id: str, action: str = "prediction") -> bool:
        """Record user action and increment usage"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check if user can perform action
        validation = self.validate_user(user_id)
        if not validation['valid']:
            return False
        
        # Increment usage
        user['usage'] += 1
        user['last_used'] = datetime.now().isoformat()
        
        # Add to usage history
        user['usage_history'].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'usage_count': user['usage']
        })
        
        # Auto-suspend if limit reached
        if user['usage'] >= user['monthly_limit']:
            user['status'] = 'suspended'
        
        self.save_users()
        return True
    
    def reset_monthly_usage(self, user_id: str = None):
        """Reset usage for specific user or all users"""
        if user_id:
            if user_id in self.users:
                self.users[user_id]['usage'] = 0
                self.users[user_id]['status'] = 'active'
        else:
            for user in self.users.values():
                user['usage'] = 0
                user['status'] = 'active'
        
        self.save_users()
    
    def get_user_stats(self) -> Dict:
        """Get comprehensive user statistics"""
        total_users = len(self.users)
        active_users = sum(1 for u in self.users.values() if u['status'] == 'active')
        suspended_users = sum(1 for u in self.users.values() if u['status'] == 'suspended')
        premium_users = sum(1 for u in self.users.values() if u['tier'] == 'premium')
        total_usage = sum(u['usage'] for u in self.users.values())
        avg_usage = total_usage / total_users if total_users > 0 else 0
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'suspended_users': suspended_users,
            'premium_users': premium_users,
            'total_usage': total_usage,
            'avg_usage': round(avg_usage, 1)
        }
    
    def export_credentials(self) -> str:
        """Export user credentials as CSV string"""
        data = []
        for user in self.users.values():
            data.append({
                'User ID': user['id'],
                'Name': user['name'],
                'Email': user['email'],
                'API Key': user['api_key'],
                'Tier': user['tier'],
                'Monthly Limit': user['monthly_limit'],
                'Current Usage': user['usage'],
                'Status': user['status'],
                'Created': user['created'][:10]  # Just date part
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)


def create_user_management_section():
    """User Management Section - Add this to your main app"""
    st.header("👥 User Management System")
    
    # Initialize user manager
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    
    user_manager = st.session_state.user_manager
    
    # Get current stats
    stats = user_manager.get_user_stats()
    
    # Stats Dashboard
    st.subheader("📊 User Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Users", stats['total_users'])
    with col2:
        st.metric("Active Users", stats['active_users'])
    with col3:
        st.metric("Suspended", stats['suspended_users'])
    with col4:
        st.metric("Premium Users", stats['premium_users'])
    with col5:
        st.metric("Avg Usage", stats['avg_usage'])
    
    st.markdown("---")
    
    # Management Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add New User", type="primary"):
            new_user_id = user_manager.add_user()
            st.success(f"✅ Created user: {new_user_id}")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All Usage"):
            user_manager.reset_monthly_usage()
            st.success("✅ Reset usage for all users")
            st.rerun()
    
    with col3:
        # Export credentials
        csv_data = user_manager.export_credentials()
        st.download_button(
            label="📥 Download Credentials",
            data=csv_data,
            file_name=f"user_credentials_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Users Table
    st.subheader("👤 User Management")
    
    # Convert users to DataFrame for display
    users_data = []
    for user in user_manager.users.values():
        users_data.append({
            'User ID': user['id'],
            'Name': user['name'],
            'Email': user['email'],
            'Tier': user['tier'],
            'Usage': f"{user['usage']}/{user['monthly_limit']}",
            'Status': user['status'],
            'Last Used': user['last_used'][:10] if user['last_used'] else 'Never',
            'API Key': f"{user['api_key'][:12]}..."
        })
    
    if users_data:
        df = pd.DataFrame(users_data)
        
        # Display with color coding
        def highlight_status(row):
            if row['Status'] == 'suspended':
                return ['background-color: #ffebee'] * len(row)
            elif row['Tier'] == 'premium':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: white'] * len(row)
        
        styled_df = df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Individual user management
        st.subheader("🔧 Individual User Management")
        
        selected_user_id = st.selectbox(
            "Select User to Manage",
            options=list(user_manager.users.keys()),
            format_func=lambda x: f"{x} - {user_manager.users[x]['name']}"
        )
        
        if selected_user_id:
            user = user_manager.users[selected_user_id]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Reset Usage"):
                    user_manager.reset_monthly_usage(selected_user_id)
                    st.success(f"✅ Reset usage for {selected_user_id}")
                    st.rerun()
            
            with col2:
                current_status = user['status']
                new_status = 'suspended' if current_status == 'active' else 'active'
                if st.button(f"{'🚫 Suspend' if current_status == 'active' else '✅ Activate'}"):
                    user_manager.users[selected_user_id]['status'] = new_status
                    user_manager.save_users()
                    st.success(f"✅ User {new_status}")
                    st.rerun()
            
            with col3:
                new_limit = st.number_input(
                    "Monthly Limit", 
                    value=user['monthly_limit'], 
                    min_value=1, 
                    max_value=1000,
                    key=f"limit_{selected_user_id}"
                )
                if st.button("💾 Update Limit"):
                    user_manager.users[selected_user_id]['monthly_limit'] = new_limit
                    user_manager.save_users()
                    st.success(f"✅ Updated limit to {new_limit}")
                    st.rerun()
            
            with col4:
                if st.button("🗑️ Remove User"):
                    del user_manager.users[selected_user_id]
                    user_manager.save_users()
                    st.success(f"✅ Removed user {selected_user_id}")
                    st.rerun()
            
            # User details
            st.markdown("### 📋 User Details")
            user_details_col1, user_details_col2 = st.columns(2)
            
            with user_details_col1:
                st.write(f"**User ID:** {user['id']}")
                st.write(f"**Name:** {user['name']}")
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Tier:** {user['tier']}")
            
            with user_details_col2:
                st.write(f"**API Key:** {user['api_key']}")
                st.write(f"**Status:** {user['status']}")
                st.write(f"**Created:** {user['created'][:10]}")
                st.write(f"**Usage:** {user['usage']}/{user['monthly_limit']}")


def user_access_middleware(user_id: str) -> bool:
    """Middleware function to check user access before AI predictions"""
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    
    user_manager = st.session_state.user_manager
    validation = user_manager.validate_user(user_id)
    
    if not validation['valid']:
        st.error(f"❌ Access Denied: {validation['reason']}")
        return False
    
    # Record the usage
    user_manager.record_usage(user_id, "ai_prediction")
    
    remaining = validation['remaining_usage'] - 1
    if remaining <= 2:
        st.warning(f"⚠️ Usage Warning: Only {remaining} predictions remaining this month")
    
    return True


