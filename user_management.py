# COMPLETE USER MANAGEMENT SYSTEM
# Replace your existing user management code with this comprehensive version

import streamlit as st
import pandas as pd
import json
import secrets
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

class ProfessionalUserManager:
    """Complete Professional User Management System"""
    
    def __init__(self, data_file="users_data.json"):
        self.data_file = data_file
        self.users = self.load_users()
        
        # Enhanced user ID templates
        self.id_templates = {
            'simple': 'USER_{counter:03d}',
            'secure': 'ATPS_{random_hex}',
            'business': '{prefix}_{date}_{counter:02d}',
            'crypto': 'TRD_{crypto_hash}',
            'professional': 'PRO_{date}_{random_short}'
        }
        
        # Business prefixes for professional appearance
        self.business_prefixes = [
            'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 'OMEGA',
            'PRIME', 'ELITE', 'VIP', 'GOLD', 'PLATINUM', 'DIAMOND',
            'TRADE', 'INVEST', 'PROFIT', 'GROWTH', 'WEALTH', 'CAPITAL'
        ]
        
        # Initialize with demo users if empty
        if not self.users:
            self._create_demo_users()
    
    def load_users(self) -> Dict:
        """Load users from JSON file with error handling"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Validate data structure
                    if isinstance(data, dict):
                        return data
        except Exception as e:
            print(f"Error loading users: {e}")
        return {}
    
    def save_users(self):
        """Save users to JSON file with error handling"""
        try:
            # Create backup before saving
            if os.path.exists(self.data_file):
                backup_file = f"{self.data_file}.backup"
                with open(self.data_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            # Save current data
            with open(self.data_file, 'w') as f:
                json.dump(self.users, f, default=str, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _create_demo_users(self):
        """Create initial demo users for testing"""
        demo_users = [
            {'template': 'simple', 'count': 3, 'tier': 'free'},
            {'template': 'secure', 'count': 2, 'tier': 'premium'},
        ]
        
        for demo in demo_users:
            try:
                self.bulk_generate_users(
                    count=demo['count'],
                    template_type=demo['template'],
                    tier=demo['tier'],
                    monthly_limit=10
                )
            except Exception as e:
                print(f"Error creating demo users: {e}")
    
    def generate_secure_id(self, template_type: str = 'simple', **kwargs) -> str:
        """Generate secure user ID based on template with uniqueness guarantee"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            try:
                if template_type == 'simple':
                    counter = len(self.users) + 1
                    user_id = f"USER_{counter:03d}"
                    
                elif template_type == 'secure':
                    random_hex = secrets.token_hex(8).upper()
                    user_id = f"ATPS_{random_hex}"
                    
                elif template_type == 'business':
                    prefix = kwargs.get('prefix', 'TRADE')
                    date_str = datetime.now().strftime('%m%d')
                    counter = len([u for u in self.users.keys() if prefix in u]) + 1
                    user_id = f"{prefix}_{date_str}_{counter:02d}"
                    
                elif template_type == 'crypto':
                    random_data = f"{datetime.now().isoformat()}{secrets.token_hex(16)}"
                    crypto_hash = hashlib.sha256(random_data.encode()).hexdigest()[:12].upper()
                    user_id = f"TRD_{crypto_hash}"
                    
                elif template_type == 'professional':
                    date_str = datetime.now().strftime('%y%m%d')
                    random_short = secrets.token_hex(3).upper()
                    user_id = f"PRO_{date_str}_{random_short}"
                    
                else:
                    # Fallback to simple
                    counter = len(self.users) + 1
                    user_id = f"USER_{counter:03d}"
                
                # Check uniqueness
                if user_id not in self.users:
                    return user_id
                    
            except Exception as e:
                print(f"Error generating ID: {e}")
        
        # Fallback with timestamp if all attempts fail
        timestamp = int(datetime.now().timestamp())
        return f"USER_{timestamp}"
    
    def generate_api_key(self, format_type: str = 'standard') -> str:
        """Generate API key with different security levels"""
        if format_type == 'standard':
            return f"atps_{secrets.token_hex(16)}"
        elif format_type == 'secure':
            return f"sk_live_{secrets.token_hex(24)}"
        elif format_type == 'enterprise':
            return f"ent_{secrets.token_hex(20)}"
        else:
            return f"key_{secrets.token_hex(18)}"
    
    def validate_user(self, user_id: str) -> Dict[str, Any]:
        """Comprehensive user validation with detailed feedback"""
        if not user_id or not user_id.strip():
            return {'valid': False, 'reason': 'User ID is empty'}
        
        user_id = user_id.strip()
        
        if user_id not in self.users:
            return {
                'valid': False, 
                'reason': 'User ID not found',
                'suggestion': 'Check your User ID or contact administrator'
            }
        
        user = self.users[user_id]
        
        if user['status'] != 'active':
            return {
                'valid': False, 
                'reason': f'Account is {user["status"]}',
                'suggestion': 'Contact administrator to reactivate your account'
            }
        
        if user['usage'] >= user['monthly_limit']:
            return {
                'valid': False, 
                'reason': 'Monthly usage limit exceeded',
                'suggestion': 'Wait for next month or contact administrator for limit increase'
            }
        
        return {
            'valid': True, 
            'user': user,
            'remaining_usage': user['monthly_limit'] - user['usage']
        }
    
    def record_usage(self, user_id: str, action: str = "prediction") -> bool:
        """Record user action with comprehensive tracking"""
        validation = self.validate_user(user_id)
        if not validation['valid']:
            return False
        
        user = self.users[user_id]
        
        # Increment usage
        user['usage'] += 1
        user['last_used'] = datetime.now().isoformat()
        
        # Add to usage history with more details
        user['usage_history'].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'usage_count': user['usage'],
            'remaining': user['monthly_limit'] - user['usage']
        })
        
        # Auto-suspend if limit reached
        if user['usage'] >= user['monthly_limit']:
            user['status'] = 'suspended'
            user['suspended_reason'] = 'Monthly limit exceeded'
            user['suspended_date'] = datetime.now().isoformat()
        
        self.save_users()
        return True
    
    def bulk_generate_users(self, count: int, template_type: str = 'simple', **kwargs) -> List[str]:
        """Generate multiple users with comprehensive error handling"""
        generated_ids = []
        failed_generations = 0
        
        for i in range(count):
            try:
                user_id = self.generate_secure_id(template_type, **kwargs)
                
                # Ensure absolute uniqueness
                attempt = 0
                while user_id in self.users or user_id in generated_ids:
                    attempt += 1
                    if attempt > 10:
                        # Add random suffix if struggling with uniqueness
                        user_id = f"{user_id}_{secrets.token_hex(2).upper()}"
                        break
                    user_id = self.generate_secure_id(template_type, **kwargs)
                
                # Create comprehensive user profile
                user_count = len(self.users) + len(generated_ids) + 1
                
                user_profile = {
                    'id': user_id,
                    'name': kwargs.get('name', f'User {user_count}'),
                    'email': kwargs.get('email', f'user{user_count}@example.com'),
                    'usage': 0,
                    'monthly_limit': kwargs.get('monthly_limit', 10),
                    'status': 'active',
                    'tier': kwargs.get('tier', 'free'),
                    'created': datetime.now().isoformat(),
                    'last_used': None,
                    'usage_history': [],
                    'api_key': self.generate_api_key(kwargs.get('api_format', 'standard')),
                    'generation_method': template_type,
                    'created_by': 'admin',
                    'notes': kwargs.get('notes', ''),
                    'custom_fields': kwargs.get('custom_fields', {}),
                    'suspended_reason': None,
                    'suspended_date': None
                }
                
                self.users[user_id] = user_profile
                generated_ids.append(user_id)
                
            except Exception as e:
                print(f"Failed to generate user {i+1}: {e}")
                failed_generations += 1
        
        self.save_users()
        
        if failed_generations > 0:
            print(f"Warning: {failed_generations} user generations failed")
        
        return generated_ids
    
    def reset_monthly_usage(self, user_id: str = None):
        """Reset usage with comprehensive tracking"""
        reset_count = 0
        reset_time = datetime.now().isoformat()
        
        if user_id:
            if user_id in self.users:
                user = self.users[user_id]
                old_usage = user['usage']
                user['usage'] = 0
                user['status'] = 'active'
                user['suspended_reason'] = None
                user['suspended_date'] = None
                user['last_reset'] = reset_time
                user['usage_history'].append({
                    'timestamp': reset_time,
                    'action': 'usage_reset',
                    'old_usage': old_usage,
                    'new_usage': 0
                })
                reset_count = 1
        else:
            for user in self.users.values():
                old_usage = user['usage']
                user['usage'] = 0
                user['status'] = 'active'
                user['suspended_reason'] = None
                user['suspended_date'] = None
                user['last_reset'] = reset_time
                user['usage_history'].append({
                    'timestamp': reset_time,
                    'action': 'bulk_usage_reset',
                    'old_usage': old_usage,
                    'new_usage': 0
                })
                reset_count += 1
        
        self.save_users()
        return reset_count
    
    def get_comprehensive_stats(self) -> Dict:
        """Get detailed system statistics"""
        total_users = len(self.users)
        if total_users == 0:
            return {'total_users': 0}
        
        active_users = sum(1 for u in self.users.values() if u['status'] == 'active')
        suspended_users = sum(1 for u in self.users.values() if u['status'] == 'suspended')
        premium_users = sum(1 for u in self.users.values() if u['tier'] == 'premium')
        total_usage = sum(u['usage'] for u in self.users.values())
        avg_usage = total_usage / total_users
        
        # Generation method stats
        generation_methods = {}
        for user in self.users.values():
            method = user.get('generation_method', 'unknown')
            generation_methods[method] = generation_methods.get(method, 0) + 1
        
        # Usage distribution
        usage_ranges = {'0': 0, '1-5': 0, '6-10': 0, '10+': 0}
        for user in self.users.values():
            usage = user['usage']
            if usage == 0:
                usage_ranges['0'] += 1
            elif usage <= 5:
                usage_ranges['1-5'] += 1
            elif usage <= 10:
                usage_ranges['6-10'] += 1
            else:
                usage_ranges['10+'] += 1
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'suspended_users': suspended_users,
            'premium_users': premium_users,
            'free_users': total_users - premium_users,
            'total_usage': total_usage,
            'avg_usage': round(avg_usage, 1),
            'generation_methods': generation_methods,
            'usage_distribution': usage_ranges,
            'last_updated': datetime.now().isoformat()
        }
    
    def export_credentials(self, include_api_keys: bool = False) -> str:
        """Export user credentials with security options"""
        data = []
        for user in self.users.values():
            row = {
                'User ID': user['id'],
                'Name': user['name'],
                'Email': user['email'],
                'Tier': user['tier'],
                'Monthly Limit': user['monthly_limit'],
                'Current Usage': user['usage'],
                'Status': user['status'],
                'Generation Method': user.get('generation_method', 'unknown'),
                'Created': user['created'][:10],
                'Last Used': user['last_used'][:10] if user['last_used'] else 'Never'
            }
            
            if include_api_keys:
                row['API Key'] = user['api_key']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def search_users(self, query: str) -> List[Dict]:
        """Search users by ID, name, or email"""
        query = query.lower().strip()
        if not query:
            return list(self.users.values())
        
        results = []
        for user in self.users.values():
            if (query in user['id'].lower() or 
                query in user['name'].lower() or 
                query in user['email'].lower()):
                results.append(user)
        
        return results


def create_professional_user_management():
    """Professional User Management Interface - Complete System"""
    
    st.header("👥 Professional User Management System")
    st.caption("Complete user access control and analytics platform")
    
    # Initialize user manager
    if 'professional_user_manager' not in st.session_state:
        st.session_state.professional_user_manager = ProfessionalUserManager()
    
    user_manager = st.session_state.professional_user_manager
    
    # Get comprehensive statistics
    stats = user_manager.get_comprehensive_stats()
    
    # ========================================================================
    # 1. COMPREHENSIVE DASHBOARD
    # ========================================================================
    st.subheader("📊 System Dashboard")
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Users", stats['total_users'])
    with col2:
        st.metric("Active Users", stats['active_users'], 
                 delta=stats['active_users'] - stats['suspended_users'])
    with col3:
        st.metric("Premium Users", stats['premium_users'])
    with col4:
        st.metric("Total Usage", stats['total_usage'])
    with col5:
        st.metric("Avg Usage", stats['avg_usage'])
    
    # Secondary metrics
    if stats['total_users'] > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Suspended", stats['suspended_users'])
        with col2:
            st.metric("Free Users", stats['free_users'])
        with col3:
            active_rate = (stats['active_users'] / stats['total_users']) * 100
            st.metric("Active Rate", f"{active_rate:.1f}%")
        with col4:
            premium_rate = (stats['premium_users'] / stats['total_users']) * 100
            st.metric("Premium Rate", f"{premium_rate:.1f}%")
    
    st.markdown("---")
    
    # ========================================================================
    # 2. USER GENERATION SYSTEM
    # ========================================================================
    st.subheader("🎯 User Generation Center")
    
    gen_col1, gen_col2 = st.columns(2)
    
    with gen_col1:
        st.markdown("#### ⚙️ Generation Settings")
        
        template_type = st.selectbox(
            "User ID Template",
            options=['simple', 'secure', 'business', 'crypto', 'professional'],
            format_func=lambda x: {
                'simple': '🔢 Simple (USER_001, USER_002)',
                'secure': '🔒 Secure (ATPS_A1B2C3D4E5F6)',
                'business': '🏢 Business (ALPHA_0818_01)',
                'crypto': '💎 Crypto (TRD_A1B2C3D4E5F6)',
                'professional': '⭐ Professional (PRO_250818_A1B)'
            }[x],
            help="Choose the format for generated User IDs"
        )
        
        # Template-specific options
        if template_type == 'business':
            prefix = st.selectbox(
                "Business Prefix",
                options=user_manager.business_prefixes,
                help="Professional prefix for business clients"
            )
        else:
            prefix = None
        
        api_format = st.selectbox(
            "API Key Security Level",
            options=['standard', 'secure', 'enterprise'],
            format_func=lambda x: {
                'standard': '📝 Standard (atps_...)',
                'secure': '🔐 Secure (sk_live_...)',
                'enterprise': '🏢 Enterprise (ent_...)'
            }[x]
        )
    
    with gen_col2:
        st.markdown("#### 👥 User Configuration")
        
        generation_mode = st.radio(
            "Generation Mode",
            options=['single', 'bulk'],
            format_func=lambda x: {'single': '👤 Single User', 'bulk': '👥 Bulk Generation'}[x]
        )
        
        if generation_mode == 'bulk':
            count = st.number_input("Number of Users", min_value=1, max_value=100, value=5)
        else:
            count = 1
        
        tier = st.selectbox("Default Tier", options=['free', 'premium'], index=0)
        monthly_limit = st.number_input("Monthly Prediction Limit", min_value=1, max_value=1000, value=10)
    
    # Live Preview
    st.markdown("#### 👀 Live Preview")
    try:
        kwargs = {'monthly_limit': monthly_limit, 'tier': tier, 'api_format': api_format}
        if prefix:
            kwargs['prefix'] = prefix
        
        preview_id = user_manager.generate_secure_id(template_type, **kwargs)
        preview_api = user_manager.generate_api_key(api_format)
        
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.code(f"User ID: {preview_id}", language=None)
        with preview_col2:
            st.code(f"API Key: {preview_api[:20]}...", language=None)
    except Exception as e:
        st.error(f"Preview error: {e}")
    
    # Generation Actions
    st.markdown("#### 🚀 Generate Users")
    
    gen_action_col1, gen_action_col2, gen_action_col3 = st.columns(3)
    
    with gen_action_col1:
        if st.button("➕ Generate Users", type="primary", key="main_generate_pro"):
            with st.spinner(f"Generating {count} user(s)..."):
                try:
                    kwargs = {
                        'monthly_limit': monthly_limit, 
                        'tier': tier, 
                        'api_format': api_format,
                        'notes': f'Generated on {datetime.now().strftime("%Y-%m-%d")}'
                    }
                    if prefix:
                        kwargs['prefix'] = prefix
                    
                    generated_ids = user_manager.bulk_generate_users(count, template_type, **kwargs)
                    
                    if generated_ids:
                        st.success(f"✅ Successfully generated {len(generated_ids)} user(s)!")
                        
                        # Show generated IDs
                        with st.expander("📋 Generated User IDs (Click to copy)", expanded=True):
                            for i, user_id in enumerate(generated_ids, 1):
                                st.code(f"{i}. {user_id}")
                        
                        st.rerun()
                    else:
                        st.error("❌ Generation failed")
                        
                except Exception as e:
                    st.error(f"Generation error: {e}")
    
    with gen_action_col2:
        if st.button("⚡ Quick Secure (5 users)", key="quick_secure_pro"):
            with st.spinner("Generating 5 secure users..."):
                try:
                    generated_ids = user_manager.bulk_generate_users(
                        5, 'secure', 
                        monthly_limit=monthly_limit, 
                        tier=tier, 
                        api_format='secure'
                    )
                    st.success(f"✅ Generated {len(generated_ids)} secure users!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Quick generation failed: {e}")
    
    with gen_action_col3:
        if st.button("💎 Premium Crypto (3 users)", key="quick_crypto_pro"):
            with st.spinner("Generating 3 premium crypto users..."):
                try:
                    generated_ids = user_manager.bulk_generate_users(
                        3, 'crypto', 
                        monthly_limit=50, 
                        tier='premium', 
                        api_format='enterprise'
                    )
                    st.success(f"✅ Generated {len(generated_ids)} premium crypto users!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Crypto generation failed: {e}")
    
    st.markdown("---")
    
    # ========================================================================
    # 3. COMPREHENSIVE USERS TABLE
    # ========================================================================
    st.subheader("👤 All Users Overview")
    
    if user_manager.users:
        # Search functionality
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input(
                "🔍 Search Users", 
                placeholder="Search by User ID, name, or email...",
                key="user_search"
            )
        
        with search_col2:
            show_suspended = st.checkbox("Show Suspended", value=True)
        
        # Get filtered users
        if search_query:
            filtered_users = user_manager.search_users(search_query)
        else:
            filtered_users = list(user_manager.users.values())
        
        if not show_suspended:
            filtered_users = [u for u in filtered_users if u['status'] == 'active']
        
        # Create users table
        if filtered_users:
            users_data = []
            for user in filtered_users:
                # Calculate usage percentage
                usage_percent = (user['usage'] / user['monthly_limit']) * 100 if user['monthly_limit'] > 0 else 0
                
                users_data.append({
                    'User ID': user['id'],
                    'Name': user['name'],
                    'Tier': user['tier'].title(),
                    'Usage': f"{user['usage']}/{user['monthly_limit']}",
                    'Usage %': f"{usage_percent:.0f}%",
                    'Status': user['status'].title(),
                    'Method': user.get('generation_method', 'unknown').title(),
                    'Created': user['created'][:10],
                    'Last Used': user['last_used'][:10] if user['last_used'] else 'Never'
                })
            
            df = pd.DataFrame(users_data)
            
            # Apply color coding
            def highlight_rows(row):
                if 'Suspended' in row['Status']:
                    return ['background-color: #ffebee'] * len(row)
                elif 'Premium' in row['Tier']:
                    return ['background-color: #fff3e0'] * len(row)
                elif int(row['Usage %'].rstrip('%')) >= 80:
                    return ['background-color: #ffeaa7'] * len(row)
                else:
                    return ['background-color: white'] * len(row)
            
            styled_df = df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Show filtered results count
            if search_query or not show_suspended:
                st.caption(f"Showing {len(filtered_users)} of {len(user_manager.users)} total users")
                
        else:
            st.info("No users match your search criteria")
    
    else:
        st.info("👆 No users found. Use the generation tools above to create users!")
    
    st.markdown("---")
    
    # ========================================================================
    # 4. BULK MANAGEMENT ACTIONS
    # ========================================================================
    if user_manager.users:
        st.subheader("🔧 Bulk Management")
        
        bulk_col1, bulk_col2, bulk_col3, bulk_col4 = st.columns(4)
        
        with bulk_col1:
            if st.button("🔄 Reset All Usage", key="reset_all_pro"):
                reset_count = user_manager.reset_monthly_usage()
                st.success(f"✅ Reset usage for {reset_count} users!")
                st.rerun()
        
        with bulk_col2:
            selected_users = st.multiselect(
                "Select Users for Bulk Action",
                options=list(user_manager.users.keys()),
                format_func=lambda x: f"{x} ({user_manager.users[x]['name']})",
                key="bulk_select_pro"
            )
        
        with bulk_col3:
            if selected_users and st.button("🚫 Suspend Selected", key="suspend_bulk_pro"):
                for user_id in selected_users:
                    user_manager.users[user_id]['status'] = 'suspended'
                    user_manager.users[user_id]['suspended_reason'] = 'Admin action'
                    user_manager.users[user_id]['suspended_date'] = datetime.now().isoformat()
                user_manager.save_users()
                st.success(f"✅ Suspended {len(selected_users)} users")
                st.rerun()
        
        with bulk_col4:
            if selected_users and st.button("✅ Activate Selected", key="activate_bulk_pro"):
                for user_id in selected_users:
                    user_manager.users[user_id]['status'] = 'active'
                    user_manager.users[user_id]['suspended_reason'] = None
                    user_manager.users[user_id]['suspended_date'] = None
                user_manager.save_users()
                st.success(f"✅ Activated {len(selected_users)} users")
                st.rerun()
    
    st.markdown("---")
    
    # ========================================================================
    # 5. EXPORT AND DOWNLOAD CENTER
    # ========================================================================
    st.subheader("📥 Export & Download Center")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Standard credentials export
        csv_data = user_manager.export_credentials(include_api_keys=False)
        st.download_button(
            label="📄 Download User Credentials",
            data=csv_data,
            file_name=f"user_credentials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download User IDs and basic info (safe to share)"
        )
    
    with export_col2:
        # Complete export with API keys
        if st.button("🔐 Export with API Keys"):
            csv_data_full = user_manager.export_credentials(include_api_keys=True)
            st.download_button(
                label="⬇️ Download Complete Export",
                data=csv_data_full,
                file_name=f"complete_user_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Includes API Keys - Keep secure!"
            )
    
    with export_col3:
        # System backup
        if st.button("💾 Create System Backup"):
            backup_data = json.dumps(user_manager.users, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Backup",
                data=backup_data,
                file_name=f"user_system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Complete system backup in JSON format"
            )
    
    st.markdown("---")
    
    # ========================================================================
    # 6. INDIVIDUAL USER MANAGEMENT
    # ========================================================================
    if user_manager.users:
        st.subheader("👤 Individual User Management")
        
        selected_user_id = st.selectbox(
            "Select User to Manage",
            options=list(user_manager.users.keys()),
            format_func=lambda x: f"{x} - {user_manager.users[x]['name']} ({user_manager.users[x]['tier']})",
            key="individual_user_select"
        )
        
        if selected_user_id:
            user = user_manager.users[selected_user_id]
            
            # User Profile Display
            st.markdown("### 📋 User Profile")
            
            profile_col1, profile_col2 = st.columns(2)
            
            with profile_col1:
                st.markdown("#### 👤 Basic Information")
                st.info(f"""
                **User ID:** {user['id']}
                **Name:** {user['name']}
                **Email:** {user['email']}
                **Tier:** {user['tier'].title()}
                **Status:** {user['status'].title()}
                **Generation Method:** {user.get('generation_method', 'Unknown').title()}
                """)
                
                # Edit basic info
                with st.expander("✏️ Edit Basic Information"):
                    new_name = st.text_input("Name", value=user['name'], key=f"edit_name_{selected_user_id}")
                    new_email = st.text_input("Email", value=user['email'], key=f"edit_email_{selected_user_id}")
                    new_tier = st.selectbox("Tier", options=['free', 'premium'], 
                                          index=0 if user['tier'] == 'free' else 1, 
                                          key=f"edit_tier_{selected_user_id}")
                    
                    if st.button("💾 Save Changes", key=f"save_basic_{selected_user_id}"):
                        user_manager.users[selected_user_id]['name'] = new_name
                        user_manager.users[selected_user_id]['email'] = new_email
                        user_manager.users[selected_user_id]['tier'] = new_tier
                        user_manager.save_users()
                        st.success("✅ Basic information updated!")
                        st.rerun()
            
            with profile_col2:
                st.markdown("#### 📊 Usage Statistics")
                
                # Usage progress bar
                usage_percent = (user['usage'] / user['monthly_limit']) * 100 if user['monthly_limit'] > 0 else 0
                st.progress(usage_percent / 100)
                
                st.info(f"""
                **Current Usage:** {user['usage']} / {user['monthly_limit']} ({usage_percent:.1f}%)
                **Remaining:** {user['monthly_limit'] - user['usage']} predictions
                **Created:** {user['created'][:10]}
                **Last Used:** {user['last_used'][:10] if user['last_used'] else 'Never'}
                **Last Reset:** {user.get('last_reset', 'Never')[:10] if user.get('last_reset') else 'Never'}
                """)
                
                # Suspension info if applicable
                if user['status'] == 'suspended':
                    st.warning(f"""
                    **⚠️ Account Suspended**
                    **Reason:** {user.get('suspended_reason', 'Unknown')}
                    **Date:** {user.get('suspended_date', 'Unknown')[:10] if user.get('suspended_date') else 'Unknown'}
                    """)
            
            # Security Information
            st.markdown("#### 🔑 Security Information")
            security_col1, security_col2 = st.columns(2)
            
            with security_col1:
                with st.expander("🔑 API Key (Click to reveal)", expanded=False):
                    st.code(user['api_key'])
                    st.caption("⚠️ Keep this secure - treat like a password")
                    
                    if st.button("🔄 Regenerate API Key", key=f"regen_api_{selected_user_id}"):
                        new_api_key = user_manager.generate_api_key('standard')
                        user_manager.users[selected_user_id]['api_key'] = new_api_key
                        user_manager.save_users()
                        st.success("✅ API Key regenerated!")
                        st.rerun()
            
            with security_col2:
                st.markdown("**Account Security:**")
                st.write(f"• User ID: `{user['id']}`")
                st.write(f"• Creation Date: {user['created'][:10]}")
                st.write(f"• Total Usage History: {len(user.get('usage_history', []))} records")
                
                if user.get('notes'):
                    st.write(f"• Admin Notes: {user['notes']}")
            
            # Usage History
            if user.get('usage_history'):
                st.markdown("#### 📈 Usage History")
                
                history_data = []
                for record in user['usage_history'][-10:]:  # Last 10 records
                    history_data.append({
                        'Date': record['timestamp'][:10],
                        'Time': record['timestamp'][11:19],
                        'Action': record['action'].replace('_', ' ').title(),
                        'Usage Count': record.get('usage_count', 'N/A'),
                        'Remaining': record.get('remaining', 'N/A')
                    })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("No usage history available")
            
            # Quick Actions
            st.markdown("#### ⚙️ Quick Actions")
            
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                if st.button("🔄 Reset Usage", key=f"reset_individual_{selected_user_id}"):
                    reset_count = user_manager.reset_monthly_usage(selected_user_id)
                    st.success(f"✅ Usage reset for {selected_user_id}")
                    st.rerun()
            
            with action_col2:
                current_status = user['status']
                new_status = 'suspended' if current_status == 'active' else 'active'
                action_text = '🚫 Suspend' if current_status == 'active' else '✅ Activate'
                
                if st.button(action_text, key=f"toggle_status_{selected_user_id}"):
                    user_manager.users[selected_user_id]['status'] = new_status
                    if new_status == 'suspended':
                        user_manager.users[selected_user_id]['suspended_reason'] = 'Admin action'
                        user_manager.users[selected_user_id]['suspended_date'] = datetime.now().isoformat()
                    else:
                        user_manager.users[selected_user_id]['suspended_reason'] = None
                        user_manager.users[selected_user_id]['suspended_date'] = None
                    
                    user_manager.save_users()
                    st.success(f"✅ User {new_status}")
                    st.rerun()
            
            with action_col3:
                new_limit = st.number_input(
                    "New Monthly Limit", 
                    value=user['monthly_limit'], 
                    min_value=1, 
                    max_value=1000,
                    key=f"new_limit_{selected_user_id}"
                )
                if st.button("💾 Update Limit", key=f"update_limit_{selected_user_id}"):
                    old_limit = user_manager.users[selected_user_id]['monthly_limit']
                    user_manager.users[selected_user_id]['monthly_limit'] = new_limit
                    
                    # Add to history
                    user_manager.users[selected_user_id]['usage_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'limit_update',
                        'old_limit': old_limit,
                        'new_limit': new_limit
                    })
                    
                    user_manager.save_users()
                    st.success(f"✅ Monthly limit updated to {new_limit}")
                    st.rerun()
            
            with action_col4:
                if st.button("🗑️ Remove User", key=f"remove_{selected_user_id}", type="secondary"):
                    if st.session_state.get(f'confirm_remove_{selected_user_id}', False):
                        del user_manager.users[selected_user_id]
                        user_manager.save_users()
                        st.success(f"✅ User {selected_user_id} removed")
                        st.rerun()
                    else:
                        st.session_state[f'confirm_remove_{selected_user_id}'] = True
                        st.warning("⚠️ Click again to confirm permanent removal")
            
            # Admin Notes
            st.markdown("#### 📝 Admin Notes")
            current_notes = user.get('notes', '')
            new_notes = st.text_area(
                "Notes (visible only to admins)",
                value=current_notes,
                height=100,
                key=f"notes_{selected_user_id}"
            )
            
            if new_notes != current_notes:
                if st.button("💾 Save Notes", key=f"save_notes_{selected_user_id}"):
                    user_manager.users[selected_user_id]['notes'] = new_notes
                    user_manager.save_users()
                    st.success("✅ Notes saved!")
                    st.rerun()


# ========================================================================
# USER ACCESS MIDDLEWARE - ENHANCED VERSION
# ========================================================================

def enhanced_user_access_middleware(user_id: str) -> bool:
    """Enhanced middleware function with comprehensive validation and tracking"""
    if 'professional_user_manager' not in st.session_state:
        st.session_state.professional_user_manager = ProfessionalUserManager()
    
    user_manager = st.session_state.professional_user_manager
    validation = user_manager.validate_user(user_id)
    
    if not validation['valid']:
        st.error(f"❌ **Access Denied:** {validation['reason']}")
        
        # Show helpful suggestions
        if 'suggestion' in validation:
            st.info(f"💡 **Suggestion:** {validation['suggestion']}")
        
        return False
    
    # Record the usage
    success = user_manager.record_usage(user_id, "ai_prediction")
    
    if success:
        remaining = validation['remaining_usage'] - 1
        
        # Show usage warnings
        if remaining <= 2 and remaining > 0:
            st.warning(f"⚠️ **Usage Warning:** Only {remaining} predictions remaining this month")
        elif remaining <= 0:
            st.error("🚫 **Limit Reached:** This was your last prediction for this month")
        
        return True
    else:
        st.error("❌ **Usage Recording Failed:** Please try again")
        return False


# ========================================================================
# INTEGRATION FUNCTIONS
# ========================================================================

def create_user_management_section():
    """Main function to call the professional user management system"""
    create_professional_user_management()


def user_access_middleware(user_id: str) -> bool:
    """Backward compatibility function"""
    return enhanced_user_access_middleware(user_id)


# ========================================================================
# EXAMPLE DISTRIBUTION EMAIL TEMPLATES
# ========================================================================

def generate_user_email_template(user_id: str, user_name: str, app_url: str) -> str:
    """Generate professional email template for user distribution"""
    
    template = f"""
Subject: Your AI Trading Platform Access - {user_id}

Dear {user_name},

Welcome to our AI Trading Platform! Your account has been created and is ready to use.

🔐 Your Credentials:
   User ID: {user_id}
   Monthly Predictions: 10
   Account Status: Active

🚀 Getting Started:
   1. Visit: {app_url}
   2. Enter "{user_id}" in the sidebar under "User Access"
   3. You'll see a welcome message and usage tracker
   4. Go to the "Prediction" tab to start making AI predictions

📊 Your Account:
   • You get 10 AI predictions per month
   • Usage resets automatically each month
   • Your predictions are tracked in real-time
   • Contact support if you need assistance

🔒 Security:
   • Keep your User ID secure
   • Do not share with others
   • Report any issues immediately

📞 Support:
   If you have any questions or need assistance, please reply to this email.

Best regards,
AI Trading Platform Team

---
This is an automated message. Please do not reply to this email directly.
"""
    
    return template.strip()


# ========================================================================
# ADMIN HELPER FUNCTIONS
# ========================================================================

def get_user_distribution_summary(user_manager) -> str:
    """Generate a summary for admin distribution tracking"""
    
    total_users = len(user_manager.users)
    active_users = sum(1 for u in user_manager.users.values() if u['status'] == 'active')
    premium_users = sum(1 for u in user_manager.users.values() if u['tier'] == 'premium')
    
    summary = f"""
📊 User Distribution Summary:
   • Total Users: {total_users}
   • Active Users: {active_users}
   • Premium Users: {premium_users}
   • Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📋 User IDs Ready for Distribution:
"""
    
    for user_id, user in user_manager.users.items():
        summary += f"   • {user_id} → {user['name']} ({user['tier']})\n"
    
    return summary
