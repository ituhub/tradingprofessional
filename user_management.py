# Enhanced user_management.py - Replace your existing UserManager class with this enhanced version

import streamlit as st
import pandas as pd
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid
import hashlib
import os
import secrets
import string
import random
from pathlib import Path

class EnhancedUserManager:
    """Enhanced User Management System with Advanced User Generation"""
    
    def __init__(self, data_file="users_data.json"):
        self.data_file = data_file
        self.users = self.load_users()
        
        # User ID generation templates
        self.id_templates = {
            'simple': 'USER_{counter:03d}',
            'secure': 'ATPS_{random_hex}',
            'business': '{prefix}_{date}_{counter:02d}',
            'crypto': 'TRD_{crypto_hash}',
            'custom': '{custom_format}'
        }
        
        # Predefined prefixes for business format
        self.business_prefixes = [
            'TRADE', 'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 
            'PRIME', 'ELITE', 'PRO', 'VIP', 'GOLD', 'PLATINUM'
        ]
        
    def load_users(self) -> Dict:
        """Load users from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading users: {e}")
        
        return {}
    
    def save_users(self):
        """Save users to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.users, f, default=str, indent=2)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def generate_secure_id(self, template_type: str = 'simple', **kwargs) -> str:
        """Generate secure user ID based on template"""
        
        if template_type == 'simple':
            counter = len(self.users) + 1
            return f"USER_{counter:03d}"
            
        elif template_type == 'secure':
            # Generate cryptographically secure random hex
            random_hex = secrets.token_hex(8).upper()
            return f"ATPS_{random_hex}"
            
        elif template_type == 'business':
            prefix = kwargs.get('prefix', random.choice(self.business_prefixes))
            date_str = datetime.now().strftime('%m%d')
            counter = len([u for u in self.users.keys() if prefix in u]) + 1
            return f"{prefix}_{date_str}_{counter:02d}"
            
        elif template_type == 'crypto':
            # Generate crypto-style hash
            random_data = f"{datetime.now().isoformat()}{secrets.token_hex(16)}"
            crypto_hash = hashlib.sha256(random_data.encode()).hexdigest()[:12].upper()
            return f"TRD_{crypto_hash}"
            
        elif template_type == 'custom':
            custom_format = kwargs.get('custom_format', 'USER_{counter:03d}')
            counter = len(self.users) + 1
            random_hex = secrets.token_hex(4).upper()
            date_str = datetime.now().strftime('%Y%m%d')
            
            # Replace placeholders
            user_id = custom_format.format(
                counter=counter,
                random_hex=random_hex,
                date=date_str,
                timestamp=int(datetime.now().timestamp())
            )
            return user_id
            
        else:
            # Fallback to simple
            return self.generate_secure_id('simple')
    
    def generate_api_key(self, format_type: str = 'standard') -> str:
        """Generate API key in different formats"""
        
        if format_type == 'standard':
            return f"atps_{secrets.token_hex(16)}"
        elif format_type == 'secure':
            return f"sk_live_{secrets.token_hex(24)}"
        elif format_type == 'jwt_style':
            # JWT-style with dots
            part1 = secrets.token_urlsafe(8)
            part2 = secrets.token_urlsafe(16)
            part3 = secrets.token_urlsafe(8)
            return f"{part1}.{part2}.{part3}"
        else:
            return f"key_{secrets.token_hex(20)}"
    
    def bulk_generate_users(self, count: int, template_type: str = 'simple', **kwargs) -> List[str]:
        """Generate multiple users at once"""
        generated_ids = []
        
        for i in range(count):
            user_id = self.generate_secure_id(template_type, **kwargs)
            
            # Ensure uniqueness
            while user_id in self.users or user_id in generated_ids:
                user_id = self.generate_secure_id(template_type, **kwargs)
            
            # Create user
            self.users[user_id] = {
                'id': user_id,
                'name': f'User {len(self.users) + 1}',
                'email': f'user{len(self.users) + 1}@example.com',
                'usage': 0,
                'monthly_limit': kwargs.get('monthly_limit', 10),
                'status': 'active',
                'tier': kwargs.get('tier', 'free'),
                'created': datetime.now().isoformat(),
                'last_used': None,
                'usage_history': [],
                'api_key': self.generate_api_key(kwargs.get('api_format', 'standard')),
                'generation_method': template_type,
                'custom_fields': kwargs.get('custom_fields', {})
            }
            
            generated_ids.append(user_id)
        
        self.save_users()
        return generated_ids
    
    def add_single_user(self, template_type: str = 'simple', name: str = None, 
                       email: str = None, **kwargs) -> str:
        """Add a single user with custom parameters"""
        
        user_id = self.generate_secure_id(template_type, **kwargs)
        
        # Ensure uniqueness
        while user_id in self.users:
            user_id = self.generate_secure_id(template_type, **kwargs)
        
        user_count = len(self.users) + 1
        
        self.users[user_id] = {
            'id': user_id,
            'name': name or f'User {user_count}',
            'email': email or f'user{user_count}@example.com',
            'usage': 0,
            'monthly_limit': kwargs.get('monthly_limit', 10),
            'status': 'active',
            'tier': kwargs.get('tier', 'free'),
            'created': datetime.now().isoformat(),
            'last_used': None,
            'usage_history': [],
            'api_key': self.generate_api_key(kwargs.get('api_format', 'standard')),
            'generation_method': template_type,
            'custom_fields': kwargs.get('custom_fields', {})
        }
        
        self.save_users()
        return user_id
    
    # Keep all your existing methods (validate_user, record_usage, etc.)
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
        
        # Generation method stats
        generation_methods = {}
        for user in self.users.values():
            method = user.get('generation_method', 'unknown')
            generation_methods[method] = generation_methods.get(method, 0) + 1
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'suspended_users': suspended_users,
            'premium_users': premium_users,
            'total_usage': total_usage,
            'avg_usage': round(avg_usage, 1),
            'generation_methods': generation_methods
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
                'Generation Method': user.get('generation_method', 'unknown'),
                'Created': user['created'][:10]  # Just date part
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)


def create_enhanced_user_generation_interface():
    """Enhanced user generation interface"""
    st.subheader("🎯 Advanced User Generation")
    
    # Initialize enhanced user manager
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
    # Generation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Generation Settings")
        
        template_type = st.selectbox(
            "User ID Template",
            options=['simple', 'secure', 'business', 'crypto', 'custom'],
            format_func=lambda x: {
                'simple': '🔢 Simple (USER_001, USER_002)',
                'secure': '🔒 Secure (ATPS_A1B2C3D4)',
                'business': '🏢 Business (ALPHA_1215_01)',
                'crypto': '💎 Crypto (TRD_A1B2C3D4E5F6)',
                'custom': '⚙️ Custom Format'
            }[x]
        )
        
        # Template-specific options
        if template_type == 'business':
            prefix = st.selectbox(
                "Business Prefix",
                options=user_manager.business_prefixes,
                index=0
            )
        elif template_type == 'custom':
            custom_format = st.text_input(
                "Custom Format",
                value="TRADE_{date}_{counter:03d}",
                help="Use {counter}, {random_hex}, {date}, {timestamp}"
            )
        
        api_format = st.selectbox(
            "API Key Format",
            options=['standard', 'secure', 'jwt_style'],
            format_func=lambda x: {
                'standard': '📝 Standard (atps_...)',
                'secure': '🔐 Secure (sk_live_...)',
                'jwt_style': '🎫 JWT Style (xxx.yyy.zzz)'
            }[x]
        )
    
    with col2:
        st.markdown("#### 👥 User Settings")
        
        generation_mode = st.radio(
            "Generation Mode",
            options=['single', 'bulk'],
            format_func=lambda x: {
                'single': '👤 Single User',
                'bulk': '👥 Bulk Generation'
            }[x]
        )
        
        if generation_mode == 'bulk':
            bulk_count = st.number_input(
                "Number of Users",
                min_value=1,
                max_value=100,
                value=5
            )
        
        default_tier = st.selectbox(
            "Default Tier",
            options=['free', 'premium'],
            index=0
        )
        
        monthly_limit = st.number_input(
            "Monthly Limit",
            min_value=1,
            max_value=1000,
            value=10
        )
    
    # Preview section
    st.markdown("#### 👀 Preview")
    
    # Generate preview
    preview_kwargs = {
        'monthly_limit': monthly_limit,
        'tier': default_tier,
        'api_format': api_format
    }
    
    if template_type == 'business':
        preview_kwargs['prefix'] = prefix
    elif template_type == 'custom':
        preview_kwargs['custom_format'] = custom_format
    
    try:
        preview_id = user_manager.generate_secure_id(template_type, **preview_kwargs)
        preview_api = user_manager.generate_api_key(api_format)
        
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"User ID: {preview_id}")
        with col2:
            st.code(f"API Key: {preview_api[:20]}...")
    except Exception as e:
        st.error(f"Preview error: {e}")
    
    # Generation buttons
    st.markdown("#### 🚀 Generate Users")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Generate Users", type="primary"):
            try:
                if generation_mode == 'single':
                    user_id = user_manager.add_single_user(
                        template_type=template_type,
                        **preview_kwargs
                    )
                    st.success(f"✅ Created user: {user_id}")
                else:
                    generated_ids = user_manager.bulk_generate_users(
                        count=bulk_count,
                        template_type=template_type,
                        **preview_kwargs
                    )
                    st.success(f"✅ Generated {len(generated_ids)} users!")
                    
                    # Show generated IDs
                    with st.expander("📋 Generated User IDs"):
                        for user_id in generated_ids:
                            st.code(user_id)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
    
    with col2:
        # Quick templates
        if st.button("⚡ Quick Secure (5 users)"):
            try:
                generated_ids = user_manager.bulk_generate_users(
                    count=5,
                    template_type='secure',
                    monthly_limit=monthly_limit,
                    tier=default_tier,
                    api_format='secure'
                )
                st.success(f"✅ Generated 5 secure users!")
                with st.expander("📋 Generated Secure IDs"):
                    for user_id in generated_ids:
                        st.code(user_id)
                st.rerun()
            except Exception as e:
                st.error(f"Quick generation failed: {e}")
    
    with col3:
        if st.button("💎 Crypto Style (3 users)"):
            try:
                generated_ids = user_manager.bulk_generate_users(
                    count=3,
                    template_type='crypto',
                    monthly_limit=monthly_limit,
                    tier='premium',  # Crypto users get premium
                    api_format='jwt_style'
                )
                st.success(f"✅ Generated 3 crypto-style users!")
                with st.expander("📋 Generated Crypto IDs"):
                    for user_id in generated_ids:
                        st.code(user_id)
                st.rerun()
            except Exception as e:
                st.error(f"Crypto generation failed: {e}")


def create_user_management_section():
    """Enhanced user management section with advanced generation"""
    st.header("👥 Advanced User Management System")
    
    # Initialize enhanced user manager
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
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
    
    # Generation methods breakdown
    if stats['generation_methods']:
        st.markdown("**Generation Methods:**")
        method_cols = st.columns(len(stats['generation_methods']))
        for i, (method, count) in enumerate(stats['generation_methods'].items()):
            with method_cols[i]:
                st.metric(method.title(), count)
    
    st.markdown("---")
    
    # Enhanced User Generation Interface
    create_enhanced_user_generation_interface()
    
    st.markdown("---")
    
    # Management Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All Usage"):
            user_manager.reset_monthly_usage()
            st.success("✅ Reset usage for all users")
            st.rerun()
    
    with col2:
        # Export credentials
        csv_data = user_manager.export_credentials()
        st.download_button(
            label="📥 Download All Credentials",
            data=csv_data,
            file_name=f"user_credentials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("🗑️ Clear All Users"):
            if st.session_state.get('confirm_clear', False):
                user_manager.users = {}
                user_manager.save_users()
                st.success("✅ All users cleared")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion of ALL users")
    
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
            'Method': user.get('generation_method', 'unknown'),
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
            elif 'secure' in row['Method'] or 'crypto' in row['Method']:
                return ['background-color: #e8f5e8'] * len(row)
            else:
                return ['background-color: white'] * len(row)
        
        styled_df = df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Individual user management
        st.subheader("🔧 Individual User Management")
        
        selected_user_id = st.selectbox(
            "Select User to Manage",
            options=list(user_manager.users.keys()),
            format_func=lambda x: f"{x} - {user_manager.users[x]['name']} ({user_manager.users[x].get('generation_method', 'unknown')})"
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
                st.write(f"**Generation Method:** {user.get('generation_method', 'unknown')}")
            
            with user_details_col2:
                st.write(f"**API Key:** {user['api_key']}")
                st.write(f"**Status:** {user['status']}")
                st.write(f"**Created:** {user['created'][:10]}")
                st.write(f"**Usage:** {user['usage']}/{user['monthly_limit']}")
                st.write(f"**Last Used:** {user['last_used'][:10] if user['last_used'] else 'Never'}")
    
    else:
        st.info("👆 No users found. Use the generation tools above to create users!")


# Keep your existing user_access_middleware function unchanged
def user_access_middleware(user_id: str) -> bool:
    """Middleware function to check user access before AI predictions"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
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
