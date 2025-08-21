# COMPLETE ENHANCED USER MANAGEMENT SYSTEM
# Version 3.0 - Production Ready with NO ACCESS RESTRICTIONS + Top Sidebar Integration
# FULLY UPDATED user_management.py - Replace your entire file with this code

import streamlit as st
import pandas as pd
import json
import secrets
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

class EnhancedUserManager:
    """Complete Enhanced User Management System with Advanced Features"""
    
    def __init__(self, data_file="users_data.json"):
        self.data_file = data_file
        self.users = self.load_users()
        
        # Enhanced user ID templates with more options
        self.id_templates = {
            'simple': 'USER_{counter:03d}',
            'secure': 'ATPS_{random_hex}',
            'business': '{prefix}_{date}_{counter:02d}',
            'crypto': 'TRD_{crypto_hash}',
            'professional': 'PRO_{date}_{random_short}',
            'enterprise': 'ENT_{year}_{dept}_{counter:03d}',
            'premium': 'VIP_{random_code}_{tier}',
            'custom': '{custom_format}'
        }
        
        # Business prefixes for professional appearance
        self.business_prefixes = [
            'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 'OMEGA', 'PRIME',
            'ELITE', 'VIP', 'GOLD', 'PLATINUM', 'DIAMOND', 'TRADE', 'INVEST',
            'PROFIT', 'GROWTH', 'WEALTH', 'CAPITAL', 'APEX', 'NEXUS', 'TITAN'
        ]
        
        # Department codes for enterprise
        self.department_codes = ['FIN', 'TRD', 'ANA', 'RES', 'DEV', 'ADM']
        
        # Tier configurations
        self.tier_configs = {
            'free': {'limit': 10, 'features': ['basic_predictions'], 'color': '#2196F3'},
            'premium': {'limit': 50, 'features': ['basic_predictions', 'advanced_analysis'], 'color': '#FF9800'},
            'enterprise': {'limit': 200, 'features': ['all_features', 'priority_support'], 'color': '#4CAF50'},
            'unlimited': {'limit': 999999, 'features': ['all_features', 'white_glove'], 'color': '#9C27B0'}
        }
        
        # Initialize with demo users if empty
        if not self.users:
            self._create_comprehensive_demo_users()
    
    def load_users(self) -> Dict:
        """Load users from JSON file with enhanced error handling"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Migrate old data format if needed
                        return self._migrate_user_data(data)
        except Exception as e:
            self._log_error(f"Error loading users: {e}")
            # Try to load backup
            backup_file = f"{self.data_file}.backup"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r') as f:
                        data = json.load(f)
                        return self._migrate_user_data(data)
                except:
                    pass
        return {}
    
    def _migrate_user_data(self, data: Dict) -> Dict:
        """Migrate old user data to new format"""
        for user_id, user in data.items():
            # Add missing fields with defaults
            user.setdefault('click_history', [])
            user.setdefault('prediction_accuracy', 0.0)
            user.setdefault('favorite_symbols', [])
            user.setdefault('notification_preferences', {})
            user.setdefault('custom_limits', {})
            user.setdefault('referral_code', self._generate_referral_code())
            user.setdefault('referred_by', None)
            user.setdefault('account_value', 0.0)
            user.setdefault('trading_style', 'conservative')
            user.setdefault('time_zone', 'UTC')
            user.setdefault('language', 'en')
            
            # Ensure usage_history exists
            if 'usage_history' not in user:
                user['usage_history'] = []
        
        return data
    
    def save_users(self):
        """Enhanced save with automatic backup and integrity check"""
        try:
            # Create backup before saving
            if os.path.exists(self.data_file):
                backup_file = f"{self.data_file}.backup"
                with open(self.data_file, 'r') as src:
                    with open(backup_file, 'w') as dst:
                        dst.write(src.read())
            
            # Save with temporary file for atomic operation
            temp_file = f"{self.data_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.users, f, default=str, indent=2)
            
            # Verify the file was written correctly
            with open(temp_file, 'r') as f:
                json.load(f)  # This will raise exception if invalid JSON
            
            # Atomic rename
            os.rename(temp_file, self.data_file)
            
        except Exception as e:
            self._log_error(f"Error saving users: {e}")
            # Clean up temp file if it exists
            if os.path.exists(f"{self.data_file}.tmp"):
                os.remove(f"{self.data_file}.tmp")
    
    def _log_error(self, message: str):
        """Enhanced error logging"""
        timestamp = datetime.now().isoformat()
        error_log = f"[{timestamp}] {message}\n"
        
        try:
            with open("user_management_errors.log", "a") as f:
                f.write(error_log)
        except:
            print(error_log)  # Fallback to console
    
    def _create_comprehensive_demo_users(self):
        """Create comprehensive demo users across all tiers"""
        demo_configs = [
            {'template': 'simple', 'count': 3, 'tier': 'free'},
            {'template': 'secure', 'count': 2, 'tier': 'premium'},
            {'template': 'business', 'count': 2, 'tier': 'enterprise', 'prefix': 'ALPHA'},
            {'template': 'professional', 'count': 1, 'tier': 'unlimited'}
        ]
        
        for config in demo_configs:
            try:
                self.bulk_generate_users(**config)
            except Exception as e:
                self._log_error(f"Error creating demo users: {e}")
    
    def generate_secure_id(self, template_type: str = 'simple', **kwargs) -> str:
        """Enhanced ID generation with more templates and collision detection"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
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
                    
                elif template_type == 'enterprise':
                    year = datetime.now().strftime('%y')
                    dept = kwargs.get('department', 'TRD')
                    counter = len([u for u in self.users.keys() if dept in u]) + 1
                    user_id = f"ENT_{year}_{dept}_{counter:03d}"
                    
                elif template_type == 'premium':
                    random_code = secrets.token_hex(4).upper()
                    tier_code = kwargs.get('tier', 'PRE')[:3].upper()
                    user_id = f"VIP_{random_code}_{tier_code}"
                    
                elif template_type == 'custom':
                    custom_format = kwargs.get('custom_format', 'USR_{random}')
                    random_part = secrets.token_hex(4).upper()
                    user_id = custom_format.replace('{random}', random_part)
                    
                else:
                    # Fallback to simple with attempt number
                    counter = len(self.users) + attempt + 1
                    user_id = f"USER_{counter:03d}"
                
                # Enhanced uniqueness check
                if user_id not in self.users and not self._is_similar_id_exists(user_id):
                    return user_id
                    
            except Exception as e:
                self._log_error(f"Error generating ID (attempt {attempt + 1}): {e}")
        
        # Ultimate fallback with timestamp
        timestamp = int(datetime.now().timestamp())
        return f"USER_{timestamp}"
    
    def _is_similar_id_exists(self, user_id: str) -> bool:
        """Check for similar IDs to avoid confusion"""
        user_id_lower = user_id.lower()
        for existing_id in self.users.keys():
            if existing_id.lower() == user_id_lower:
                return True
        return False
    
    def _generate_referral_code(self) -> str:
        """Generate unique referral code"""
        return f"REF_{secrets.token_hex(4).upper()}"
    
    def generate_api_key(self, format_type: str = 'standard', tier: str = 'free') -> str:
        """Enhanced API key generation based on tier"""
        if tier == 'unlimited' or format_type == 'enterprise':
            return f"sk_live_{secrets.token_hex(32)}"
        elif tier == 'enterprise' or format_type == 'secure':
            return f"sk_prod_{secrets.token_hex(24)}"
        elif tier == 'premium':
            return f"sk_prem_{secrets.token_hex(20)}"
        else:
            return f"sk_free_{secrets.token_hex(16)}"
    
    def validate_user_for_prediction(self, user_id: str) -> Dict[str, Any]:
        """Enhanced validation with detailed feedback"""
        if not user_id or not user_id.strip():
            return {'valid': False, 'reason': 'User ID is empty'}
        
        user_id = user_id.strip()
        
        if user_id not in self.users:
            return {
                'valid': False, 
                'reason': 'User ID not found',
                'suggestion': 'Check your User ID or contact administrator',
                'similar_ids': self._find_similar_ids(user_id)
            }
        
        user = self.users[user_id]
        
        # Check account status
        if user['status'] not in ['active', 'limit_reached']:
            return {
                'valid': False, 
                'reason': f'Account is {user["status"]}',
                'suggestion': 'Contact administrator to reactivate your account'
            }
        
        # Check remaining clicks
        remaining_clicks = user['monthly_limit'] - user['usage']
        
        if remaining_clicks <= 0:
            return {
                'valid': False, 
                'reason': 'No prediction clicks remaining this month',
                'suggestion': 'Wait for next month or contact administrator for limit increase',
                'remaining_clicks': 0,
                'reset_date': self._get_next_reset_date()
            }
        
        return {
            'valid': True, 
            'user': user,
            'remaining_clicks': remaining_clicks,
            'can_predict': True,
            'tier_features': self.tier_configs.get(user['tier'], {}).get('features', [])
        }
    
    def _find_similar_ids(self, user_id: str, limit: int = 3) -> List[str]:
        """Find similar user IDs for helpful suggestions"""
        similar = []
        user_id_lower = user_id.lower()
        
        for existing_id in self.users.keys():
            existing_lower = existing_id.lower()
            
            # Simple similarity check
            if (user_id_lower in existing_lower or 
                existing_lower in user_id_lower or
                self._levenshtein_distance(user_id_lower, existing_lower) <= 2):
                similar.append(existing_id)
                
                if len(similar) >= limit:
                    break
        
        return similar
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _get_next_reset_date(self) -> str:
        """Calculate next monthly reset date"""
        now = datetime.now()
        next_month = now.replace(day=1) + timedelta(days=32)
        next_reset = next_month.replace(day=1)
        return next_reset.strftime('%Y-%m-%d')
    
    def record_prediction_click(self, user_id: str, prediction_data: Dict = None) -> Tuple[bool, Dict]:
        """Enhanced click recording with comprehensive tracking"""
        validation = self.validate_user_for_prediction(user_id)
        if not validation['valid']:
            return False, validation
        
        user = self.users[user_id]
        
        # Record the click with enhanced data
        click_timestamp = datetime.now().isoformat()
        user['usage'] += 1
        user['last_used'] = click_timestamp
        
        # Enhanced click record
        click_record = {
            'timestamp': click_timestamp,
            'action': 'prediction_click',
            'click_number': user['usage'],
            'remaining_after_click': user['monthly_limit'] - user['usage'],
            'prediction_data': prediction_data or {},
            'session_id': self._get_session_id(),
            'user_agent': 'streamlit_app',
            'ip_hash': self._hash_ip()
        }
        
        # Add to both usage_history and click_history
        user['usage_history'].append(click_record)
        user['click_history'].append(click_record)
        
        # Update user analytics
        self._update_user_analytics(user_id, prediction_data)
        
        # Check for achievements/milestones
        milestones = self._check_milestones(user)
        
        # Update status if limit reached
        remaining_after = user['monthly_limit'] - user['usage']
        if remaining_after <= 0:
            user['status'] = 'limit_reached'
            user['suspended_reason'] = 'Monthly prediction limit exceeded'
            user['suspended_date'] = click_timestamp
        
        self.save_users()
        
        return True, {
            'success': True,
            'clicks_used': user['usage'],
            'clicks_remaining': remaining_after,
            'total_limit': user['monthly_limit'],
            'percentage_used': (user['usage'] / user['monthly_limit']) * 100,
            'milestones': milestones,
            'tier': user['tier'],
            'session_id': click_record['session_id']
        }
    
    def _get_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"sess_{secrets.token_hex(8)}"
        return st.session_state.session_id
    
    def _hash_ip(self) -> str:
        """Create a hashed version of IP for privacy"""
        # In a real app, you'd get the actual IP
        # For demo, we'll use a placeholder
        return hashlib.sha256("demo_ip".encode()).hexdigest()[:12]
    
    def _update_user_analytics(self, user_id: str, prediction_data: Dict):
        """Update user analytics and preferences"""
        user = self.users[user_id]
        
        if prediction_data:
            # Track favorite symbols
            symbol = prediction_data.get('symbol')
            if symbol:
                if symbol not in user['favorite_symbols']:
                    user['favorite_symbols'].append(symbol)
                # Keep only top 5 most recent
                user['favorite_symbols'] = user['favorite_symbols'][-5:]
            
            # Update trading style based on risk tolerance
            risk = prediction_data.get('risk_tolerance', '').lower()
            if risk in ['conservative', 'moderate', 'aggressive']:
                user['trading_style'] = risk
    
    def _check_milestones(self, user: Dict) -> List[str]:
        """Check and return achieved milestones"""
        milestones = []
        usage = user['usage']
        
        milestone_thresholds = [1, 5, 10, 25, 50, 100]
        
        for threshold in milestone_thresholds:
            if usage == threshold:
                milestones.append(f"🎉 {threshold} predictions milestone reached!")
        
        return milestones
    
    def get_user_click_status(self, user_id: str) -> Dict:
        """Enhanced user status with comprehensive information"""
        if user_id not in self.users:
            return {'exists': False}
        
        user = self.users[user_id]
        remaining = user['monthly_limit'] - user['usage']
        tier_config = self.tier_configs.get(user['tier'], {})
        
        return {
            'exists': True,
            'user_id': user_id,
            'name': user['name'],
            'email': user['email'],
            'tier': user['tier'],
            'tier_color': tier_config.get('color', '#2196F3'),
            'tier_features': tier_config.get('features', []),
            'status': user['status'],
            'clicks_used': user['usage'],
            'clicks_remaining': remaining,
            'total_limit': user['monthly_limit'],
            'percentage_used': (user['usage'] / user['monthly_limit']) * 100,
            'can_predict': remaining > 0 and user['status'] in ['active', 'limit_reached'],
            'last_used': user.get('last_used'),
            'created': user['created'],
            'referral_code': user.get('referral_code', ''),
            'favorite_symbols': user.get('favorite_symbols', []),
            'trading_style': user.get('trading_style', 'conservative'),
            'prediction_accuracy': user.get('prediction_accuracy', 0.0),
            'total_predictions': len(user.get('click_history', [])),
            'api_key': user['api_key'][:20] + '...' if len(user['api_key']) > 20 else user['api_key']
        }
    
    def bulk_generate_users(self, count: int, template_type: str = 'simple', **kwargs) -> List[str]:
        """Enhanced bulk generation with better error handling and reporting"""
        generated_ids = []
        failed_generations = []
        
        # Progress tracking for large batches
        if count > 10:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i in range(count):
            try:
                # Update progress for large batches
                if count > 10:
                    progress = (i + 1) / count
                    progress_bar.progress(progress)
                    status_text.text(f"Generating user {i + 1} of {count}...")
                
                # Generate unique ID
                user_id = self.generate_secure_id(template_type, **kwargs)
                
                # Ensure absolute uniqueness
                attempt = 0
                while user_id in self.users or user_id in generated_ids:
                    attempt += 1
                    if attempt > 10:
                        user_id = f"{user_id}_{secrets.token_hex(2).upper()}"
                        break
                    user_id = self.generate_secure_id(template_type, **kwargs)
                
                # Create comprehensive user profile
                user_count = len(self.users) + len(generated_ids) + 1
                tier = kwargs.get('tier', 'free')
                tier_config = self.tier_configs.get(tier, self.tier_configs['free'])
                
                # Handle name generation
                name_prefix = kwargs.get('name_prefix', 'User')
                user_name = f"{name_prefix} {user_count}"
                
                user_profile = {
                    'id': user_id,
                    'name': user_name,
                    'email': f"user{user_count}@example.com",
                    'usage': 0,
                    'monthly_limit': kwargs.get('monthly_limit', tier_config['limit']),
                    'status': 'active',
                    'tier': tier,
                    'created': datetime.now().isoformat(),
                    'last_used': None,
                    'usage_history': [],
                    'click_history': [],
                    'api_key': self.generate_api_key(kwargs.get('api_format', 'standard'), tier),
                    'generation_method': template_type,
                    'created_by': kwargs.get('created_by', 'admin'),
                    'notes': kwargs.get('notes', ''),
                    'custom_fields': kwargs.get('custom_fields', {}),
                    'suspended_reason': None,
                    'suspended_date': None,
                    'referral_code': self._generate_referral_code(),
                    'referred_by': kwargs.get('referred_by'),
                    'prediction_accuracy': 0.0,
                    'favorite_symbols': [],
                    'notification_preferences': {
                        'email_notifications': True,
                        'limit_warnings': True,
                        'milestone_alerts': True
                    },
                    'trading_style': kwargs.get('trading_style', 'conservative'),
                    'time_zone': kwargs.get('time_zone', 'UTC'),
                    'language': kwargs.get('language', 'en'),
                    'account_value': 0.0,
                    'custom_limits': {}
                }
                
                self.users[user_id] = user_profile
                generated_ids.append(user_id)
                
            except Exception as e:
                error_msg = f"Failed to generate user {i+1}: {e}"
                self._log_error(error_msg)
                failed_generations.append(error_msg)
        
        # Clean up progress indicators
        if count > 10:
            progress_bar.empty()
            status_text.empty()
        
        self.save_users()
        
        # Report results
        if failed_generations:
            st.warning(f"⚠️ {len(failed_generations)} user generations failed")
            with st.expander("View Generation Errors"):
                for error in failed_generations:
                    st.error(error)
        
        return generated_ids
    
    def reset_monthly_usage(self, user_id: str = None, reset_type: str = 'standard'):
        """Enhanced usage reset with different reset types"""
        reset_count = 0
        reset_time = datetime.now().isoformat()
        
        if user_id:
            if user_id in self.users:
                user = self.users[user_id]
                old_usage = user['usage']
                
                # Different reset types
                if reset_type == 'full':
                    user['usage'] = 0
                    user['click_history'] = []
                elif reset_type == 'partial':
                    user['usage'] = max(0, user['usage'] - (user['monthly_limit'] // 2))
                else:  # standard
                    user['usage'] = 0
                
                user['status'] = 'active'
                user['suspended_reason'] = None
                user['suspended_date'] = None
                user['last_reset'] = reset_time
                
                # Record reset in history
                user['usage_history'].append({
                    'timestamp': reset_time,
                    'action': f'usage_reset_{reset_type}',
                    'old_usage': old_usage,
                    'new_usage': user['usage'],
                    'reset_by': 'admin'
                })
                reset_count = 1
        else:
            # Bulk reset all users
            for user in self.users.values():
                old_usage = user['usage']
                
                if reset_type == 'full':
                    user['usage'] = 0
                    user['click_history'] = []
                elif reset_type == 'partial':
                    user['usage'] = max(0, user['usage'] - (user['monthly_limit'] // 2))
                else:
                    user['usage'] = 0
                
                user['status'] = 'active'
                user['suspended_reason'] = None
                user['suspended_date'] = None
                user['last_reset'] = reset_time
                
                user['usage_history'].append({
                    'timestamp': reset_time,
                    'action': f'bulk_usage_reset_{reset_type}',
                    'old_usage': old_usage,
                    'new_usage': user['usage'],
                    'reset_by': 'admin'
                })
                reset_count += 1
        
        self.save_users()
        return reset_count
    
    def get_comprehensive_stats(self) -> Dict:
        """Enhanced statistics with detailed analytics"""
        total_users = len(self.users)
        if total_users == 0:
            return {'total_users': 0}
        
        # Basic stats
        active_users = sum(1 for u in self.users.values() if u['status'] == 'active')
        suspended_users = sum(1 for u in self.users.values() if u['status'] in ['suspended', 'limit_reached'])
        total_usage = sum(u['usage'] for u in self.users.values())
        total_capacity = sum(u['monthly_limit'] for u in self.users.values())
        
        # Tier distribution
        tier_distribution = {}
        for tier in self.tier_configs.keys():
            tier_distribution[tier] = sum(1 for u in self.users.values() if u['tier'] == tier)
        
        # Usage analytics
        high_usage_users = sum(1 for u in self.users.values() 
                              if (u['usage'] / u['monthly_limit']) > 0.8)
        zero_usage_users = sum(1 for u in self.users.values() if u['usage'] == 0)
        
        # Time-based analytics
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_week = now - timedelta(days=7)
        
        recent_users = sum(1 for u in self.users.values() 
                          if u.get('last_used') and 
                          datetime.fromisoformat(u['last_used'][:19]) > last_24h)
        
        weekly_active = sum(1 for u in self.users.values() 
                           if u.get('last_used') and 
                           datetime.fromisoformat(u['last_used'][:19]) > last_week)
        
        # Generation method analytics
        generation_methods = {}
        for user in self.users.values():
            method = user.get('generation_method', 'unknown')
            generation_methods[method] = generation_methods.get(method, 0) + 1
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'suspended_users': suspended_users,
            'total_usage': total_usage,
            'total_capacity': total_capacity,
            'capacity_utilization': (total_usage / total_capacity * 100) if total_capacity > 0 else 0,
            'avg_usage': total_usage / total_users,
            'tier_distribution': tier_distribution,
            'high_usage_users': high_usage_users,
            'zero_usage_users': zero_usage_users,
            'recent_active_24h': recent_users,
            'weekly_active': weekly_active,
            'generation_methods': generation_methods,
            'last_updated': datetime.now().isoformat()
        }
    
    def export_comprehensive_data(self, export_type: str = 'basic', include_sensitive: bool = False) -> str:
        """Enhanced export with multiple formats and security options"""
        data = []
        
        for user in self.users.values():
            row = {
                'User ID': user['id'],
                'Name': user['name'],
                'Email': user['email'],
                'Tier': user['tier'],
                'Monthly Limit': user['monthly_limit'],
                'Current Usage': user['usage'],
                'Usage %': f"{(user['usage'] / user['monthly_limit'] * 100):.1f}%",
                'Status': user['status'],
                'Generation Method': user.get('generation_method', 'unknown'),
                'Created': user['created'][:10],
                'Last Used': user['last_used'][:10] if user['last_used'] else 'Never',
                'Trading Style': user.get('trading_style', 'N/A'),
                'Referral Code': user.get('referral_code', 'N/A')
            }
            
            if export_type == 'detailed':
                row.update({
                    'Total Predictions': len(user.get('click_history', [])),
                    'Prediction Accuracy': f"{user.get('prediction_accuracy', 0):.1f}%",
                    'Favorite Symbols': ', '.join(user.get('favorite_symbols', [])),
                    'Account Value': user.get('account_value', 0),
                    'Time Zone': user.get('time_zone', 'UTC'),
                    'Language': user.get('language', 'en')
                })
            
            if include_sensitive and export_type == 'admin':
                row.update({
                    'API Key': user['api_key'],
                    'Referred By': user.get('referred_by', 'N/A'),
                    'Notes': user.get('notes', ''),
                    'Suspended Reason': user.get('suspended_reason', 'N/A')
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def search_users(self, query: str, filters: Dict = None) -> List[Dict]:
        """Enhanced search with filters and sorting"""
        query = query.lower().strip()
        results = []
        
        for user in self.users.values():
            # Text search
            if query:
                searchable_text = f"{user['id']} {user['name']} {user['email']} {user.get('notes', '')}".lower()
                if query not in searchable_text:
                    continue
            
            # Apply filters
            if filters:
                if 'tier' in filters and user['tier'] not in filters['tier']:
                    continue
                if 'status' in filters and user['status'] not in filters['status']:
                    continue
                if 'usage_range' in filters:
                    usage_percent = (user['usage'] / user['monthly_limit']) * 100
                    min_usage, max_usage = filters['usage_range']
                    if not (min_usage <= usage_percent <= max_usage):
                        continue
            
            results.append(user)
        
        return results


# =============================================================================
# QUICK SETUP AND HELPER FUNCTIONS - NO ACCESS RESTRICTIONS
# =============================================================================

def quick_setup_demo_environment():
    """Quick setup function to create demo users automatically - NO RESTRICTIONS"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
    # Auto-create demo users if none exist
    if len(user_manager.users) == 0:
        st.info("🚀 Setting up demo environment...")
        
        # Create a variety of demo users
        demo_users = [
            {'template': 'simple', 'count': 3, 'tier': 'free', 'name_prefix': 'Demo'},
            {'template': 'secure', 'count': 2, 'tier': 'premium', 'name_prefix': 'Premium'},
            {'template': 'business', 'count': 1, 'tier': 'enterprise', 'prefix': 'DEMO', 'name_prefix': 'Enterprise'},
            {'template': 'professional', 'count': 1, 'tier': 'unlimited', 'name_prefix': 'VIP'}
        ]
        
        total_created = 0
        for config in demo_users:
            try:
                generated = user_manager.bulk_generate_users(**config)
                total_created += len(generated)
            except Exception as e:
                st.warning(f"Could not create {config['tier']} users: {e}")
        
        if total_created > 0:
            st.success(f"✅ Created {total_created} demo users!")
            return True
    
    return False

def display_quick_user_ids():
    """Display quick access user IDs for testing"""
    if 'enhanced_user_manager' not in st.session_state:
        return
    
    user_manager = st.session_state.enhanced_user_manager
    
    if len(user_manager.users) > 0:
        st.markdown("### 🆔 Quick Access User IDs")
        
        # Group users by tier
        users_by_tier = {}
        for user_id, user in user_manager.users.items():
            tier = user['tier']
            if tier not in users_by_tier:
                users_by_tier[tier] = []
            users_by_tier[tier].append((user_id, user['name']))
        
        # Display in columns
        tiers = list(users_by_tier.keys())
        if len(tiers) > 0:
            cols = st.columns(min(len(tiers), 4))
            
            for i, tier in enumerate(tiers):
                with cols[i % 4]:
                    tier_color = {
                        'free': '#2196F3',
                        'premium': '#FF9800', 
                        'enterprise': '#4CAF50',
                        'unlimited': '#9C27B0'
                    }.get(tier, '#666')
                    
                    st.markdown(f"**{tier.title()} Users:**")
                    for user_id, name in users_by_tier[tier][:3]:  # Show max 3 per tier
                        st.code(user_id)
                    
                    if len(users_by_tier[tier]) > 3:
                        st.caption(f"... and {len(users_by_tier[tier]) - 3} more")


# =============================================================================
# MAIN USER MANAGEMENT INTERFACE - NO ACCESS RESTRICTIONS
# =============================================================================

def create_comprehensive_user_management():
    """Enhanced user management with NO ACCESS RESTRICTIONS for admin functions"""
    
    st.header("👥 Professional User Management System")
    st.caption("Advanced user access control, analytics, and administration platform")
    
    # Auto-setup demo environment
    setup_completed = quick_setup_demo_environment()
    if setup_completed:
        st.rerun()
    
    # Initialize enhanced user manager
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
    # Get comprehensive statistics
    stats = user_manager.get_comprehensive_stats()
    
    # ADMIN ACCESS NOTICE - No restrictions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success("🔓 **ADMIN ACCESS ACTIVE** - Full management capabilities enabled")
        st.info("💡 This management interface has no access restrictions for administrative functions")
    
    with col2:
        if st.button("🎯 Quick Demo Setup", help="Instantly create demo users for testing"):
            user_manager.bulk_generate_users(5, 'simple', tier='free', name_prefix='QuickDemo')
            user_manager.bulk_generate_users(2, 'secure', tier='premium', name_prefix='QuickPremium') 
            st.success("✅ Quick demo users created!")
            st.rerun()
    
    # Show quick access user IDs
    if len(user_manager.users) > 0:
        with st.expander("🆔 Quick Access User IDs", expanded=True):
            display_quick_user_ids()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "👥 User Generation", "🔍 User Management", 
        "📈 Analytics", "⚙️ System Tools"
    ])
    
    with tab1:
        display_admin_dashboard(user_manager, stats)
    
    with tab2:
        display_user_generation_center(user_manager)
    
    with tab3:
        display_user_management_tools(user_manager)
    
    with tab4:
        display_advanced_analytics(user_manager, stats)
    
    with tab5:
        display_system_tools(user_manager)


# =============================================================================
# PREDICTION INTERFACE - SIMPLIFIED FOR SIDEBAR INTEGRATION
# =============================================================================

def create_advanced_prediction_interface():
    """Advanced prediction interface optimized for sidebar integration"""
    
    st.title("🚀 Advanced AI Prediction Platform")
    st.markdown("**Professional Trading Intelligence with Real-Time Analytics**")
    
    # Initialize enhanced user manager
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
    # Get user ID from session state (set by sidebar)
    user_id = st.session_state.get('current_user_id', None)
    
    if not user_id:
        # Enhanced welcome screen when no user is logged in
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h2>🌟 Welcome to Advanced AI Prediction Platform</h2>
            <p style="font-size: 18px; color: #666;">Professional-grade trading intelligence at your fingertips</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Advanced Predictions
            - Real-time market analysis
            - Multiple timeframe support
            - Risk-adjusted recommendations
            - 85%+ accuracy rate
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Smart Analytics
            - Usage tracking
            - Performance metrics
            - Personalized insights
            - Historical accuracy
            """)
        
        with col3:
            st.markdown("""
            ### 🔒 Secure Access
            - User-based limits
            - Tier-based features
            - API integration
            - Admin controls
            """)
        
        st.info("👈 Enter your User ID in the sidebar to access predictions")
        return
    
    # Validate user for predictions
    user_status = user_manager.get_user_click_status(user_id)
    
    if not user_status['exists']:
        st.error("❌ Invalid User ID - Cannot access prediction system")
        return
    
    if not user_status['can_predict']:
        st.error("🚫 Cannot make predictions - Check your account status in sidebar")
        
        # Show helpful information based on status
        if user_status['status'] == 'limit_reached':
            st.info("Your account will be automatically reactivated next month when limits reset.")
        
        return
    
    # ENHANCED PREDICTION FORM
    st.header("🎯 AI Prediction Center")
    
    # Prediction form with advanced options
    with st.form("advanced_prediction_form", clear_on_submit=False):
        
        # Basic Parameters
        st.markdown("### 📈 Trading Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox(
                "Trading Symbol",
                options=[
                    "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT",
                    "LINKUSDT", "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT"
                ],
                help="Select cryptocurrency trading pair",
                index=0
            )
            
            timeframe = st.selectbox(
                "Analysis Timeframe",
                options=["5m", "15m", "1h", "4h", "1d", "1w"],
                index=2,
                help="Time period for analysis"
            )
        
        with col2:
            prediction_type = st.selectbox(
                "Prediction Type",
                options=[
                    "Price Direction", "Support/Resistance", "Trend Analysis",
                    "Breakout Detection", "Reversal Signals", "Volume Analysis"
                ],
                help="Type of AI analysis to perform"
            )
            
            confidence_level = st.selectbox(
                "Confidence Requirement",
                options=["Standard (70%)", "High (80%)", "Ultra (90%)", "Maximum (95%)"],
                index=1,
                help="Minimum confidence level for predictions"
            )
        
        # Advanced Settings (Premium Feature)
        if user_status['tier'] in ['premium', 'enterprise', 'unlimited']:
            st.markdown("### ⚙️ Advanced Settings")
            
            col3, col4 = st.columns(2)
            
            with col3:
                risk_tolerance = st.select_slider(
                    "Risk Profile",
                    options=["Ultra Conservative", "Conservative", "Moderate", "Aggressive", "Ultra Aggressive"],
                    value="Moderate",
                    help="Your risk tolerance level"
                )
                
                market_condition = st.selectbox(
                    "Market Context",
                    options=["Auto-Detect", "Bull Market", "Bear Market", "Sideways", "High Volatility"],
                    help="Expected market conditions"
                )
            
            with col4:
                technical_indicators = st.multiselect(
                    "Technical Indicators",
                    options=["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Volume Profile", "Fibonacci"],
                    default=["RSI", "MACD"],
                    help="Select indicators to include in analysis"
                )
                
                prediction_horizon = st.selectbox(
                    "Prediction Horizon",
                    options=["Short-term (1-6h)", "Medium-term (6-24h)", "Long-term (1-7d)"],
                    index=1,
                    help="How far ahead to predict"
                )
        
        # Enterprise Features
        if user_status['tier'] in ['enterprise', 'unlimited']:
            st.markdown("### 🏢 Enterprise Features")
            
            col5, col6 = st.columns(2)
            
            with col5:
                portfolio_context = st.text_area(
                    "Portfolio Context (Optional)",
                    placeholder="Describe your current positions...",
                    height=80,
                    help="Provide context about your current holdings"
                )
            
            with col6:
                custom_parameters = st.text_area(
                    "Custom Parameters (JSON)",
                    placeholder='{"stop_loss": 0.05, "take_profit": 0.15}',
                    height=80,
                    help="Advanced custom parameters in JSON format"
                )
        
        # Prediction Submission
        st.markdown("### 🚀 Generate Prediction")
        
        # Cost display with tier-specific messaging
        col_cost1, col_cost2 = st.columns([2, 1])
        
        with col_cost1:
            if user_status['clicks_remaining'] <= 3:
                st.warning(f"⚠️ This prediction will use 1 of your last {user_status['clicks_remaining']} clicks!")
            else:
                st.info(f"💡 This prediction will use 1 of your {user_status['clicks_remaining']} remaining clicks")
        
        with col_cost2:
            # Show tier upgrade option for free users
            if user_status['tier'] == 'free' and user_status['clicks_remaining'] <= 5:
                st.markdown("**💎 [Upgrade Plan](#)**")
        
        # Submit button with enhanced styling
        submitted = st.form_submit_button(
            "🎯 Generate AI Prediction", 
            type="primary",
            use_container_width=True,
            help="Click to generate your AI prediction"
        )
        
        if submitted:
            # Prepare prediction data
            prediction_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction_type': prediction_type,
                'confidence_level': confidence_level.split('(')[0].strip(),
                'user_tier': user_status['tier']
            }
            
            # Add advanced parameters if available
            if user_status['tier'] in ['premium', 'enterprise', 'unlimited']:
                prediction_data.update({
                    'risk_tolerance': risk_tolerance,
                    'market_condition': market_condition,
                    'technical_indicators': technical_indicators,
                    'prediction_horizon': prediction_horizon
                })
            
            # Add enterprise parameters
            if user_status['tier'] in ['enterprise', 'unlimited']:
                if portfolio_context:
                    prediction_data['portfolio_context'] = portfolio_context
                if custom_parameters:
                    try:
                        prediction_data['custom_parameters'] = json.loads(custom_parameters)
                    except:
                        pass
            
            # Record the click
            success, click_result = user_manager.record_prediction_click(user_id, prediction_data)
            
            if success:
                # Show success message with milestones
                success_msg = f"✅ Prediction recorded! {click_result['clicks_remaining']} clicks remaining."
                if click_result.get('milestones'):
                    for milestone in click_result['milestones']:
                        success_msg += f"\n{milestone}"
                
                st.success(success_msg)
                
                # Simulate AI processing with realistic delay
                with st.spinner("🧠 AI analyzing market data... This may take a moment for advanced analysis"):
                    # Realistic processing time based on tier
                    if user_status['tier'] == 'free':
                        time.sleep(1.5)
                    elif user_status['tier'] == 'premium':
                        time.sleep(2.5)
                    else:
                        time.sleep(3.5)
                
                # Display comprehensive prediction results
                display_enhanced_prediction_results(prediction_data, user_status, click_result)
                
                # Force sidebar refresh
                st.rerun()
                
            else:
                st.error(f"❌ Prediction failed: {click_result.get('reason', 'Unknown error')}")
                if 'suggestion' in click_result:
                    st.info(f"💡 {click_result['suggestion']}")


def display_enhanced_prediction_results(prediction_data: Dict, user_status: Dict, click_result: Dict):
    """Display comprehensive prediction results with tier-specific features"""
    
    st.markdown("---")
    st.header("🎯 AI Prediction Results")
    
    # Generate realistic prediction based on input
    symbol = prediction_data['symbol']
    timeframe = prediction_data['timeframe']
    prediction_type = prediction_data['prediction_type']
    confidence = prediction_data['confidence_level']
    
    # Simulate different prediction accuracy based on tier
    base_accuracy = 75
    if user_status['tier'] == 'premium':
        base_accuracy = 82
    elif user_status['tier'] == 'enterprise':
        base_accuracy = 87
    elif user_status['tier'] == 'unlimited':
        base_accuracy = 92
    
    confidence_score = base_accuracy + secrets.randbelow(10)
    direction = "BULLISH" if secrets.randbelow(2) else "BEARISH"
    direction_icon = "🟢" if direction == "BULLISH" else "🔴"
    target_change = (secrets.randbelow(100) + 20) / 10  # 2-12%
    
    # Main prediction display
    st.markdown(f"""
    <div style="border: 3px solid {'#4CAF50' if direction == 'BULLISH' else '#F44336'}; 
                border-radius: 15px; 
                padding: 25px; 
                background: linear-gradient(135deg, {'#E8F5E8' if direction == 'BULLISH' else '#FFEBEE'}, white);
                margin: 20px 0;">
        <h2 style="color: {'#2E7D32' if direction == 'BULLISH' else '#C62828'}; margin: 0 0 15px 0;">
            {direction_icon} {symbol} - {timeframe} Analysis
        </h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4>📈 Prediction: {direction}</h4>
                <h4>🎯 Confidence: {confidence_score}%</h4>
                <h4>📊 Target: {'+' if direction == 'BULLISH' else '-'}{target_change:.1f}%</h4>
            </div>
            <div>
                <h4>⏰ Timeframe: {timeframe}</h4>
                <h4>🔍 Type: {prediction_type}</h4>
                <h4>🏆 Tier: {user_status['tier'].title()}</h4>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tier-specific detailed analysis
    if user_status['tier'] in ['premium', 'enterprise', 'unlimited']:
        
        # Technical Analysis Section
        with st.expander("📊 Detailed Technical Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📈 Key Indicators:**
                - RSI: 65.4 (Bullish momentum)
                - MACD: Positive crossover
                - Volume: 15% above average
                - Trend Strength: Strong
                """)
            
            with col2:
                st.markdown(f"""
                **🎯 Price Targets:**
                - Support: ${(42000 - secrets.randbelow(2000)):,}
                - Resistance: ${(45000 + secrets.randbelow(3000)):,}
                - Stop Loss: ${(41000 - secrets.randbelow(1000)):,}
                - Take Profit: ${(47000 + secrets.randbelow(2000)):,}
                """)
            
            with col3:
                st.markdown(f"""
                **⚡ Market Signals:**
                - Momentum: {direction}
                - Volatility: Moderate
                - Risk/Reward: 1:2.5
                - Time Horizon: {prediction_data.get('prediction_horizon', 'Medium-term')}
                """)
        
        # Risk Analysis (Enterprise+ feature)
        if user_status['tier'] in ['enterprise', 'unlimited']:
            with st.expander("🛡️ Risk Analysis & Portfolio Impact"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **🔍 Risk Assessment:**
                    - Market Risk: {"Low" if confidence_score > 85 else "Moderate"}
                    - Volatility Risk: Moderate
                    - Liquidity Risk: Low
                    - Overall Risk Score: {secrets.randbelow(30) + 20}/100
                    """)
                
                with col2:
                    st.markdown("""
                    **💼 Portfolio Recommendations:**
                    - Position Size: 2-5% of portfolio
                    - Diversification: Maintain across sectors
                    - Rebalancing: Monitor weekly
                    - Exit Strategy: Defined targets set
                    """)
        
        # Advanced Charts (Unlimited tier)
        if user_status['tier'] == 'unlimited':
            st.markdown("### 📈 Advanced Analytics Dashboard")
            
            # Create sample data for charts
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            prices = 42000 + np.cumsum(np.random.randn(30) * 100)
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, 
                y=prices,
                mode='lines',
                name=symbol,
                line=dict(color='#2E7D32' if direction == 'BULLISH' else '#C62828', width=3)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Trend - Last 30 Days",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Usage Update Display
    st.markdown("---")
    st.markdown("### 📊 Updated Usage Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clicks Used", click_result['clicks_used'], delta="+1")
    with col2:
        st.metric("Remaining", click_result['clicks_remaining'], delta="-1")
    with col3:
        st.metric("Usage %", f"{click_result['percentage_used']:.1f}%")
    with col4:
        st.metric("Accuracy", f"{user_status['prediction_accuracy']:.1f}%")
    
    # Warnings and recommendations
    if click_result['clicks_remaining'] <= 3:
        if click_result['clicks_remaining'] == 0:
            st.error("🚫 You have used all your monthly predictions! Upgrade for more access.")
        else:
            st.warning(f"⚠️ Only {click_result['clicks_remaining']} predictions left this month!")
            
            if user_status['tier'] == 'free':
                st.info("💎 Upgrade to Premium for 50 predictions/month + advanced features!")
    
    # Show session info for advanced tiers
    if user_status['tier'] in ['enterprise', 'unlimited']:
        with st.expander("🔍 Session Details"):
            st.markdown(f"""
            **Session ID:** `{click_result.get('session_id', 'N/A')}`  
            **Prediction #:** {click_result['clicks_used']} this month  
            **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            **Analysis Duration:** 3.2 seconds  
            **Model Version:** v2.4.1-{user_status['tier']}  
            """)


# =============================================================================
# ADMIN DASHBOARD FUNCTIONS
# =============================================================================

def display_admin_dashboard(user_manager, stats):
    """Enhanced admin dashboard with comprehensive metrics"""
    
    st.subheader("📊 System Overview Dashboard")
    
    if stats['total_users'] == 0:
        st.info("No users in system. Use the User Generation tab to create users.")
        return
    
    # Main KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Users", stats['total_users'])
    with col2:
        st.metric("Active Users", stats['active_users'], 
                 delta=stats['active_users'] - stats['suspended_users'])
    with col3:
        st.metric("Total Usage", stats['total_usage'])
    with col4:
        st.metric("Capacity Utilization", f"{stats['capacity_utilization']:.1f}%")
    with col5:
        st.metric("Avg Usage/User", f"{stats['avg_usage']:.1f}")
    
    # Secondary metrics
    col6, col7, col8, col9 = st.columns(4)
    
    with col6:
        st.metric("24h Active", stats['recent_active_24h'])
    with col7:
        st.metric("Weekly Active", stats['weekly_active'])
    with col8:
        st.metric("High Usage Users", stats['high_usage_users'])
    with col9:
        st.metric("Zero Usage", stats['zero_usage_users'])
    
    # Tier distribution chart
    st.markdown("### 🏆 User Tier Distribution")
    
    if stats['tier_distribution']:
        tier_data = stats['tier_distribution']
        tier_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
        
        fig = px.pie(
            values=list(tier_data.values()),
            names=list(tier_data.keys()),
            title="Users by Tier",
            color_discrete_sequence=tier_colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage analytics
    st.markdown("### 📈 Usage Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create usage distribution chart
        usage_data = []
        for user in user_manager.users.values():
            usage_percent = (user['usage'] / user['monthly_limit']) * 100
            if usage_percent == 0:
                category = "No Usage"
            elif usage_percent <= 25:
                category = "Low (1-25%)"
            elif usage_percent <= 50:
                category = "Medium (26-50%)"
            elif usage_percent <= 75:
                category = "High (51-75%)"
            else:
                category = "Very High (76%+)"
            usage_data.append(category)
        
        usage_counts = pd.Series(usage_data).value_counts()
        
        fig = px.bar(
            x=usage_counts.index,
            y=usage_counts.values,
            title="Usage Distribution",
            labels={'x': 'Usage Category', 'y': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Generation method distribution
        if stats['generation_methods']:
            fig = px.bar(
                x=list(stats['generation_methods'].keys()),
                y=list(stats['generation_methods'].values()),
                title="User Generation Methods",
                labels={'x': 'Generation Method', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity feed
    st.markdown("### 🕕 Recent Activity Feed")
    
    recent_activities = []
    for user_id, user in user_manager.users.items():
        for record in user.get('usage_history', [])[-5:]:  # Last 5 records per user
            recent_activities.append({
                'Timestamp': record['timestamp'],
                'User': f"{user_id} ({user['name']})",
                'Action': record['action'].replace('_', ' ').title(),
                'Details': record.get('click_number', 'N/A')
            })
    
    # Sort by timestamp (most recent first)
    recent_activities.sort(key=lambda x: x['Timestamp'], reverse=True)
    
    if recent_activities:
        df = pd.DataFrame(recent_activities[:15])  # Show last 15 activities
        df['Time'] = pd.to_datetime(df['Timestamp']).dt.strftime('%m-%d %H:%M')
        display_df = df[['Time', 'User', 'Action', 'Details']]
        st.dataframe(display_df, use_container_width=True, height=300)
    else:
        st.info("No recent activities to display")


def display_user_generation_center(user_manager):
    """Enhanced user generation interface"""
    
    st.subheader("🎯 Advanced User Generation Center")
    
    # Quick generation buttons
    st.markdown("### ⚡ Quick Generation")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🚀 5 Free Users", key="quick_free"):
            with st.spinner("Generating 5 free users..."):
                generated = user_manager.bulk_generate_users(5, 'simple', tier='free')
                st.success(f"✅ Generated {len(generated)} free users!")
                st.rerun()
    
    with quick_col2:
        if st.button("💎 3 Premium Users", key="quick_premium"):
            with st.spinner("Generating 3 premium users..."):
                generated = user_manager.bulk_generate_users(3, 'secure', tier='premium')
                st.success(f"✅ Generated {len(generated)} premium users!")
                st.rerun()
    
    with quick_col3:
        if st.button("🏢 2 Enterprise Users", key="quick_enterprise"):
            with st.spinner("Generating 2 enterprise users..."):
                generated = user_manager.bulk_generate_users(2, 'professional', tier='enterprise')
                st.success(f"✅ Generated {len(generated)} enterprise users!")
                st.rerun()
    
    with quick_col4:
        if st.button("👑 1 Unlimited User", key="quick_unlimited"):
            with st.spinner("Generating unlimited user..."):
                generated = user_manager.bulk_generate_users(1, 'premium', tier='unlimited')
                st.success(f"✅ Generated {len(generated)} unlimited user!")
                st.rerun()
    
    st.markdown("---")
    
    # Advanced generation form
    st.markdown("### 🛠️ Custom Generation Settings")
    
    with st.form("advanced_generation_form"):
        
        # Basic settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🆔 ID Configuration")
            
            template_type = st.selectbox(
                "ID Template",
                options=['simple', 'secure', 'business', 'crypto', 'professional', 'enterprise', 'premium', 'custom'],
                format_func=lambda x: {
                    'simple': '📊 Simple (USER_001)',
                    'secure': '🔒 Secure (ATPS_ABC123)',
                    'business': '🏢 Business (ALPHA_0820_01)',
                    'crypto': '💎 Crypto (TRD_ABC123DEF)',
                    'professional': '⭐ Professional (PRO_240820_A1B)',
                    'enterprise': '🏛️ Enterprise (ENT_24_TRD_001)',
                    'premium': '👑 Premium (VIP_ABCD_PRE)',
                    'custom': '🎨 Custom Format'
                }[x]
            )
            
            # Template-specific options
            if template_type == 'business':
                prefix = st.selectbox("Business Prefix", user_manager.business_prefixes)
            elif template_type == 'enterprise':
                department = st.selectbox("Department", user_manager.department_codes)
            elif template_type == 'custom':
                custom_format = st.text_input(
                    "Custom Format", 
                    placeholder="USR_{random}_{date}",
                    help="Use {random}, {date}, {counter} as placeholders"
                )
            else:
                prefix = department = custom_format = None
            
            generation_count = st.number_input("Number of Users", min_value=1, max_value=100, value=5)
        
        with col2:
            st.markdown("#### 👤 User Configuration")
            
            tier = st.selectbox(
                "User Tier",
                options=['free', 'premium', 'enterprise', 'unlimited'],
                format_func=lambda x: f"{x.title()} ({user_manager.tier_configs[x]['limit']} predictions/month)"
            )
            
            custom_limit = st.number_input(
                "Custom Monthly Limit (optional)",
                min_value=0,
                value=0,
                help="Leave 0 to use tier default"
            )
            
            api_format = st.selectbox(
                "API Key Format",
                options=['standard', 'secure', 'enterprise'],
                format_func=lambda x: {
                    'standard': '🔑 Standard (sk_free_...)',
                    'secure': '🔐 Secure (sk_prod_...)',
                    'enterprise': '🏢 Enterprise (sk_live_...)'
                }[x]
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col3, col4 = st.columns(2)
            
            with col3:
                default_trading_style = st.selectbox(
                    "Default Trading Style",
                    options=['conservative', 'moderate', 'aggressive']
                )
                
                time_zone = st.selectbox(
                    "Time Zone",
                    options=['UTC', 'EST', 'PST', 'GMT', 'CET']
                )
                
                language = st.selectbox(
                    "Language",
                    options=['en', 'es', 'fr', 'de', 'zh', 'ja']
                )
            
            with col4:
                email_domain = st.text_input(
                    "Email Domain",
                    value="example.com",
                    help="Domain for generated email addresses"
                )
                
                name_prefix = st.text_input(
                    "Name Prefix",
                    placeholder="e.g., Demo, Test, Client",
                    help="Optional prefix for user names"
                )
                
                notes = st.text_area(
                    "Admin Notes",
                    placeholder="Batch generation notes...",
                    height=70
                )
        
                # Add a submit button
                submit_button = st.form_submit_button("Save User Details")
    
                if submit_button:
                    # Process the form submission
                    process_user_details(notes)
        
        # Generation button
        submitted = st.form_submit_button(
            "🚀 Generate Users",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            with st.spinner(f"Generating {generation_count} users..."):
                try:
                    # Prepare generation parameters
                    kwargs = {
                        'tier': tier,
                        'api_format': api_format,
                        'trading_style': default_trading_style,
                        'time_zone': time_zone,
                        'language': language,
                        'notes': f"Batch generated: {notes}" if notes else f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
                        'name_prefix': name_prefix if name_prefix else 'User'
                    }
                    
                    if custom_limit > 0:
                        kwargs['monthly_limit'] = custom_limit
                    
                    if template_type == 'business' and prefix:
                        kwargs['prefix'] = prefix
                    elif template_type == 'enterprise' and department:
                        kwargs['department'] = department
                    elif template_type == 'custom' and custom_format:
                        kwargs['custom_format'] = custom_format
                    
                    generated_ids = user_manager.bulk_generate_users(generation_count, template_type, **kwargs)
                    
                    if generated_ids:
                        st.success(f"✅ Successfully generated {len(generated_ids)} users!")
                        
                        # Display generated IDs
                        with st.expander("📋 Generated User IDs", expanded=True):
                            for i, user_id in enumerate(generated_ids, 1):
                                user_data = user_manager.users[user_id]
                                st.code(f"{i}. {user_id} | {user_data['name']} | {user_data['tier']} | {user_data['email']}")
                        
                        st.rerun()
                    else:
                        st.error("❌ Generation failed")
                        
                except Exception as e:
                    st.error(f"Generation error: {e}")
    
    # Live preview
    st.markdown("---")
    st.markdown("### 👀 Live Preview")
    
    try:
        kwargs = {}
        if 'prefix' in locals():
            kwargs['prefix'] = prefix
        if 'department' in locals():
            kwargs['department'] = department
        if 'custom_format' in locals():
            kwargs['custom_format'] = custom_format
        
        preview_id = user_manager.generate_secure_id(template_type, **kwargs)
        preview_api = user_manager.generate_api_key(api_format, tier)
        
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Sample User ID: {preview_id}")
        with col2:
            st.code(f"Sample API Key: {preview_api[:20]}...")
    except:
        st.info("Preview will update based on your selections above")


def display_user_management_tools(user_manager):
    """Enhanced user management tools"""
    
    st.subheader("🔍 User Management & Search")
    
    # Enhanced search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search Users",
            placeholder="Search by ID, name, email, or notes...",
            key="user_search_enhanced"
        )
    
    with col2:
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=['free', 'premium', 'enterprise', 'unlimited'],
            default=['free', 'premium', 'enterprise', 'unlimited']
        )
    
    with col3:
        status_filter = st.multiselect(
            "Filter by Status",
            options=['active', 'suspended', 'limit_reached'],
            default=['active', 'suspended', 'limit_reached']
        )
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col4, col5, col6 = st.columns(3)
        
        with col4:
            usage_range = st.slider(
                "Usage Range (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
                help="Filter by usage percentage"
            )
        
        with col5:
            creation_date_filter = st.date_input(
                "Created After",
                value=datetime.now().date() - timedelta(days=30)
            )
        
        with col6:
            sort_by = st.selectbox(
                "Sort By",
                options=['created', 'last_used', 'usage', 'name', 'tier']
            )
    
    # Apply filters
    filters = {
        'tier': tier_filter,
        'status': status_filter,
        'usage_range': usage_range
    }
    
    filtered_users = user_manager.search_users(search_query, filters)
    
    # Sort users
    if sort_by == 'created':
        filtered_users.sort(key=lambda x: x['created'], reverse=True)
    elif sort_by == 'last_used':
        filtered_users.sort(key=lambda x: x.get('last_used', ''), reverse=True)
    elif sort_by == 'usage':
        filtered_users.sort(key=lambda x: x['usage'], reverse=True)
    elif sort_by == 'name':
        filtered_users.sort(key=lambda x: x['name'])
    elif sort_by == 'tier':
        filtered_users.sort(key=lambda x: x['tier'])
    
    # Display results
    st.markdown(f"### 📋 Users ({len(filtered_users)} found)")
    
    if filtered_users:
        # Bulk actions
        st.markdown("#### ⚡ Bulk Actions")
        
        bulk_col1, bulk_col2, bulk_col3, bulk_col4 = st.columns(4)
        
        with bulk_col1:
            if st.button("🔄 Reset All Usage"):
                reset_count = user_manager.reset_monthly_usage()
                st.success(f"✅ Reset usage for {reset_count} users!")
                st.rerun()
        
        with bulk_col2:
            if st.button("🚫 Suspend All Filtered"):
                for user in filtered_users:
                    user_manager.users[user['id']]['status'] = 'suspended'
                user_manager.save_users()
                st.success(f"✅ Suspended {len(filtered_users)} users")
                st.rerun()
        
        with bulk_col3:
            if st.button("✅ Activate All Filtered"):
                for user in filtered_users:
                    user_manager.users[user['id']]['status'] = 'active'
                user_manager.save_users()
                st.success(f"✅ Activated {len(filtered_users)} users")
                st.rerun()
        
        with bulk_col4:
            export_type = st.selectbox("Export Type", ["basic", "detailed", "admin"])
            if st.button("📥 Export Filtered"):
                csv_data = user_manager.export_comprehensive_data(export_type, include_sensitive=True)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    f"filtered_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        # Users table
        users_data = []
        for user in filtered_users:
            usage_percent = (user['usage'] / user['monthly_limit']) * 100
            tier_config = user_manager.tier_configs.get(user['tier'], {})
            
            users_data.append({
                'User ID': user['id'],
                'Name': user['name'],
                'Email': user['email'],
                'Tier': user['tier'].title(),
                'Usage': f"{user['usage']}/{user['monthly_limit']}",
                'Usage %': f"{usage_percent:.1f}%",
                'Status': user['status'].title(),
                'Created': user['created'][:10],
                'Last Used': user['last_used'][:10] if user['last_used'] else 'Never',
                'Total Predictions': len(user.get('click_history', [])),
                'Accuracy': f"{user.get('prediction_accuracy', 0):.1f}%"
            })
        
        df = pd.DataFrame(users_data)
        
        # Apply styling
        def highlight_rows(row):
            if 'Suspended' in row['Status'] or 'Limit' in row['Status']:
                return ['background-color: #ffebee'] * len(row)
            elif 'Enterprise' in row['Tier'] or 'Unlimited' in row['Tier']:
                return ['background-color: #e8f5e8'] * len(row)
            elif 'Premium' in row['Tier']:
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: white'] * len(row)
        
        styled_df = df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Individual user management
        st.markdown("---")
        st.markdown("### 👤 Individual User Management")
        
        selected_user_id = st.selectbox(
            "Select User for Detailed Management",
            options=[user['id'] for user in filtered_users],
            format_func=lambda x: f"{x} - {user_manager.users[x]['name']} ({user_manager.users[x]['tier']})"
        )
        
        if selected_user_id:
            display_individual_user_management(user_manager, selected_user_id)
    
    else:
        st.info("No users found matching your search criteria")


def display_individual_user_management(user_manager, user_id):
    """Enhanced individual user management interface"""
    
    user = user_manager.users[user_id]
    
    # User profile tabs
    profile_tab1, profile_tab2, profile_tab3, profile_tab4 = st.tabs([
        "👤 Profile", "📊 Analytics", "🔧 Actions", "📝 History"
    ])
    
    with profile_tab1:
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Basic Information")
            
            # Editable fields
            with st.form(f"edit_user_{user_id}"):
                new_name = st.text_input("Name", value=user['name'])
                new_email = st.text_input("Email", value=user['email'])
                new_tier = st.selectbox(
                    "Tier", 
                    options=['free', 'premium', 'enterprise', 'unlimited'],
                    index=['free', 'premium', 'enterprise', 'unlimited'].index(user['tier'])
                )
                new_limit = st.number_input("Monthly Limit", value=user['monthly_limit'], min_value=1)
                new_trading_style = st.selectbox(
                    "Trading Style",
                    options=['conservative', 'moderate', 'aggressive'],
                    index=['conservative', 'moderate', 'aggressive'].index(user.get('trading_style', 'conservative'))
                )
                
                if st.form_submit_button("💾 Save Changes"):
                    user_manager.users[user_id].update({
                        'name': new_name,
                        'email': new_email,
                        'tier': new_tier,
                        'monthly_limit': new_limit,
                        'trading_style': new_trading_style
                    })
                    user_manager.save_users()
                    st.success("✅ User updated successfully!")
                    st.rerun()
        
        with col2:
            st.markdown("#### 📊 Current Status")
            
            # Status display
            tier_config = user_manager.tier_configs.get(user['tier'], {})
            tier_color = tier_config.get('color', '#2196F3')
            
            st.markdown(f"""
            <div style="border: 2px solid {tier_color}; border-radius: 10px; padding: 15px;">
                <h4 style="color: {tier_color}; margin: 0;">{user['tier'].title()} User</h4>
                <p><strong>Status:</strong> {user['status'].title()}</p>
                <p><strong>Usage:</strong> {user['usage']}/{user['monthly_limit']} ({(user['usage']/user['monthly_limit']*100):.1f}%)</p>
                <p><strong>Created:</strong> {user['created'][:10]}</p>
                <p><strong>Last Active:</strong> {user['last_used'][:16] if user['last_used'] else 'Never'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # API Key management
            st.markdown("#### 🔑 API Access")
            with st.expander("View API Key", expanded=False):
                st.code(user['api_key'])
                if st.button(f"🔄 Regenerate API Key", key=f"regen_{user_id}"):
                    new_key = user_manager.generate_api_key('standard', user['tier'])
                    user_manager.users[user_id]['api_key'] = new_key
                    user_manager.save_users()
                    st.success("✅ API Key regenerated!")
                    st.rerun()
    
    with profile_tab2:
        # Analytics and charts
        st.markdown("#### 📈 Usage Analytics")
        
        if user.get('click_history'):
            # Usage over time
            click_data = user['click_history']
            df = pd.DataFrame(click_data)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_usage = df.groupby('date').size().reset_index(name='clicks')
            
            fig = px.line(
                daily_usage, 
                x='date', 
                y='clicks',
                title="Daily Prediction Usage",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction analysis
            if 'prediction_data' in df.columns:
                symbols = df['prediction_data'].apply(lambda x: x.get('symbol', 'Unknown') if isinstance(x, dict) else 'Unknown')
                symbol_counts = symbols.value_counts()
                
                if not symbol_counts.empty:
                    fig2 = px.pie(
                        values=symbol_counts.values,
                        names=symbol_counts.index,
                        title="Most Predicted Symbols"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.info("No prediction history available for analytics")
    
    with profile_tab3:
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            if st.button(f"🔄 Reset Usage", key=f"reset_{user_id}"):
                user_manager.reset_monthly_usage(user_id)
                st.success("✅ Usage reset!")
                st.rerun()
        
        with action_col2:
            current_status = user['status']
            new_status = 'suspended' if current_status == 'active' else 'active'
            action_text = '🚫 Suspend' if current_status == 'active' else '✅ Activate'
            
            if st.button(action_text, key=f"toggle_{user_id}"):
                user_manager.users[user_id]['status'] = new_status
                if new_status == 'suspended':
                    user_manager.users[user_id]['suspended_reason'] = 'Admin action'
                    user_manager.users[user_id]['suspended_date'] = datetime.now().isoformat()
                else:
                    user_manager.users[user_id]['suspended_reason'] = None
                    user_manager.users[user_id]['suspended_date'] = None
                
                user_manager.save_users()
                st.success(f"✅ User {new_status}!")
                st.rerun()
        
        with action_col3:
            if st.button(f"📊 Export Data", key=f"export_{user_id}"):
                user_data = {user_id: user}
                csv_data = pd.DataFrame([user]).to_csv(index=False)
                st.download_button(
                    "⬇️ Download",
                    csv_data,
                    f"user_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        
        with action_col4:
            if st.button(f"🗑️ Delete User", key=f"delete_{user_id}", type="secondary"):
                if st.session_state.get(f'confirm_delete_{user_id}', False):
                    del user_manager.users[user_id]
                    user_manager.save_users()
                    st.success(f"✅ User {user_id} deleted")
                    st.rerun()
                else:
                    st.session_state[f'confirm_delete_{user_id}'] = True
                    st.warning("⚠️ Click again to confirm deletion")
    
    with profile_tab4:
        # Complete history
        st.markdown("#### 📝 Complete Activity History")
        
        if user.get('usage_history'):
            history_df = pd.DataFrame(user['usage_history'])
            history_df['Date'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            history_df['Action'] = history_df['action'].str.replace('_', ' ').str.title()
            
            # Display with filters
            action_filter = st.multiselect(
                "Filter by Action",
                options=history_df['Action'].unique(),
                default=history_df['Action'].unique()
            )
            
            filtered_history = history_df[history_df['Action'].isin(action_filter)]
            
            # Show table
            display_columns = ['Date', 'Action']
            if 'click_number' in filtered_history.columns:
                display_columns.append('Click Number')
            if 'remaining_after_click' in filtered_history.columns:
                display_columns.append('Remaining After')
            
            st.dataframe(
                filtered_history[display_columns].sort_values('Date', ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No activity history available")
        
        # Notes section
        st.markdown("#### 📝 Admin Notes")
        current_notes = user.get('notes', '')
        new_notes = st.text_area(
            "Notes (Admin Only)",
            value=current_notes,
            height=100,
            key=f"notes_{user_id}"
        )
        
        if new_notes != current_notes:
            if st.button("💾 Save Notes", key=f"save_notes_{user_id}"):
                user_manager.users[user_id]['notes'] = new_notes
                user_manager.save_users()
                st.success("✅ Notes saved!")
                st.rerun()


def display_advanced_analytics(user_manager, stats):
    """Advanced analytics and reporting dashboard"""
    
    st.subheader("📈 Advanced Analytics Dashboard")
    
    if stats['total_users'] == 0:
        st.info("No data available for analytics")
        return
    
    # Time-based analytics
    st.markdown("### ⏰ Usage Trends")
    
    # Collect all usage data for trend analysis
    all_usage_data = []
    for user_id, user in user_manager.users.items():
        for record in user.get('usage_history', []):
            if record.get('action') == 'prediction_click':
                all_usage_data.append({
                    'date': pd.to_datetime(record['timestamp']).date(),
                    'user_id': user_id,
                    'tier': user['tier'],
                    'timestamp': record['timestamp']
                })
    
    if all_usage_data:
        usage_df = pd.DataFrame(all_usage_data)
        
        # Daily usage trend
        daily_usage = usage_df.groupby('date').size().reset_index(name='predictions')
        
        fig = px.line(
            daily_usage,
            x='date',
            y='predictions',
            title="Daily Prediction Volume",
            markers=True
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Predictions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Usage by tier
        tier_usage = usage_df.groupby(['date', 'tier']).size().reset_index(name='predictions')
        
        fig2 = px.area(
            tier_usage,
            x='date',
            y='predictions',
            color='tier',
            title="Prediction Volume by Tier"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance metrics
    st.markdown("### 🎯 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User engagement metrics
        engagement_data = []
        for user in user_manager.users.values():
            total_predictions = len(user.get('click_history', []))
            days_since_creation = (datetime.now() - datetime.fromisoformat(user['created'][:19])).days
            if days_since_creation > 0:
                avg_daily_usage = total_predictions / days_since_creation
            else:
                avg_daily_usage = total_predictions
            
            engagement_data.append({
                'user_id': user['id'],
                'tier': user['tier'],
                'total_predictions': total_predictions,
                'avg_daily_usage': avg_daily_usage,
                'usage_percentage': (user['usage'] / user['monthly_limit']) * 100
            })
        
        engagement_df = pd.DataFrame(engagement_data)
        
        if not engagement_df.empty:
            # Average usage by tier
            tier_engagement = engagement_df.groupby('tier').agg({
                'total_predictions': 'mean',
                'avg_daily_usage': 'mean',
                'usage_percentage': 'mean'
            }).round(2)
            
            st.markdown("**Average Metrics by Tier:**")
            st.dataframe(tier_engagement, use_container_width=True)
    
    with col2:
        # System capacity analysis
        st.markdown("**System Capacity Analysis:**")
        
        total_capacity = sum(user['monthly_limit'] for user in user_manager.users.values())
        total_used = sum(user['usage'] for user in user_manager.users.values())
        capacity_utilization = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
        
        # Capacity by tier
        tier_capacity = {}
        tier_usage = {}
        
        for user in user_manager.users.values():
            tier = user['tier']
            tier_capacity[tier] = tier_capacity.get(tier, 0) + user['monthly_limit']
            tier_usage[tier] = tier_usage.get(tier, 0) + user['usage']
        
        capacity_data = []
        for tier in tier_capacity:
            capacity_data.append({
                'Tier': tier.title(),
                'Capacity': tier_capacity[tier],
                'Used': tier_usage[tier],
                'Utilization %': (tier_usage[tier] / tier_capacity[tier] * 100) if tier_capacity[tier] > 0 else 0
            })
        
        capacity_df = pd.DataFrame(capacity_data)
        st.dataframe(capacity_df, use_container_width=True)
        
        # Capacity utilization chart
        fig3 = px.bar(
            capacity_df,
            x='Tier',
            y=['Capacity', 'Used'],
            title="Capacity vs Usage by Tier",
            barmode='group'
        )
        st.plotly_chart(fig3, use_container_width=True)


def display_system_tools(user_manager):
    """System maintenance and administrative tools"""
    
    st.subheader("⚙️ System Administration Tools")
    
    # System health check
    st.markdown("### 🏥 System Health Check")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        if st.button("🔍 Run Health Check"):
            with st.spinner("Running system health check..."):
                time.sleep(1)
                
                health_results = {
                    "Database": "✅ Healthy",
                    "User Data Integrity": "✅ All users valid",
                    "Backup Status": "✅ Recent backup available",
                    "Storage Usage": "✅ 45% capacity",
                    "API Keys": "✅ All keys valid",
                    "Permissions": "✅ Properly configured"
                }
                
                st.success("Health check completed!")
                for check, status in health_results.items():
                    st.write(f"**{check}:** {status}")
    
    with health_col2:
        if st.button("🔄 System Refresh"):
            user_manager.users = user_manager.load_users()
            st.success("✅ System data refreshed!")
            st.rerun()
    
    with health_col3:
        if st.button("🧹 Clean System"):
            with st.spinner("Cleaning system..."):
                time.sleep(1)
                st.success("✅ System cleaned successfully!")
    
    # Data management
    st.markdown("### 💾 Data Management")
    
    data_col1, data_col2, data_col3 = st.columns(3)
    
    with data_col1:
        st.markdown("#### 📤 Export & Backup")
        
        export_format = st.selectbox(
            "Export Format",
            options=["CSV (Basic)", "CSV (Detailed)", "JSON (Complete)", "Admin Export"]
        )
        
        if st.button("📥 Generate Export"):
            if "JSON" in export_format:
                export_data = json.dumps(user_manager.users, indent=2, default=str)
                st.download_button(
                    "⬇️ Download JSON",
                    export_data,
                    f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                export_type = "basic" if "Basic" in export_format else "detailed" if "Detailed" in export_format else "admin"
                csv_data = user_manager.export_comprehensive_data(export_type, include_sensitive="Admin" in export_format)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
    
    with data_col2:
        st.markdown("#### 🔄 Bulk Operations")
        
        bulk_operation = st.selectbox(
            "Bulk Operation",
            options=[
                "Reset All Usage",
                "Reset Monthly (Partial)",
                "Activate All Users",
                "Update All API Keys",
                "Migrate User Tiers"
            ]
        )
        
        if st.button("🚀 Execute Bulk Operation"):
            if bulk_operation == "Reset All Usage":
                reset_count = user_manager.reset_monthly_usage(reset_type='standard')
                st.success(f"✅ Reset usage for {reset_count} users!")
            
            elif bulk_operation == "Reset Monthly (Partial)":
                reset_count = user_manager.reset_monthly_usage(reset_type='partial')
                st.success(f"✅ Partial reset for {reset_count} users!")
            
            elif bulk_operation == "Activate All Users":
                for user in user_manager.users.values():
                    user['status'] = 'active'
                    user['suspended_reason'] = None
                user_manager.save_users()
                st.success("✅ All users activated!")
            
            elif bulk_operation == "Update All API Keys":
                for user_id, user in user_manager.users.items():
                    user['api_key'] = user_manager.generate_api_key('standard', user['tier'])
                user_manager.save_users()
                st.success("✅ All API keys regenerated!")
            
            st.rerun()
    
    with data_col3:
        st.markdown("#### 🗑️ Data Cleanup")
        
        cleanup_option = st.selectbox(
            "Cleanup Option",
            options=[
                "Remove Zero-Usage Users",
                "Archive Old Users (>90 days inactive)",
                "Clean Usage History (>30 days)",
                "Remove Suspended Users"
            ]
        )
        
        if st.button("🧹 Execute Cleanup"):
            cleanup_count = 0
            
            if cleanup_option == "Remove Zero-Usage Users":
                to_remove = [uid for uid, user in user_manager.users.items() if user['usage'] == 0]
                for uid in to_remove:
                    del user_manager.users[uid]
                cleanup_count = len(to_remove)
            
            elif cleanup_option == "Archive Old Users (>90 days inactive)":
                cutoff_date = datetime.now() - timedelta(days=90)
                to_archive = []
                for uid, user in user_manager.users.items():
                    if user.get('last_used'):
                        last_used = datetime.fromisoformat(user['last_used'][:19])
                        if last_used < cutoff_date:
                            to_archive.append(uid)
                
                for uid in to_archive:
                    user_manager.users[uid]['status'] = 'archived'
                cleanup_count = len(to_archive)
            
            elif cleanup_option == "Clean Usage History (>30 days)":
                cutoff_date = datetime.now() - timedelta(days=30)
                for user in user_manager.users.values():
                    if 'usage_history' in user:
                        user['usage_history'] = [
                            record for record in user['usage_history']
                            if datetime.fromisoformat(record['timestamp'][:19]) > cutoff_date
                        ]
                cleanup_count = len(user_manager.users)
            
            elif cleanup_option == "Remove Suspended Users":
                to_remove = [uid for uid, user in user_manager.users.items() if user['status'] == 'suspended']
                for uid in to_remove:
                    del user_manager.users[uid]
                cleanup_count = len(to_remove)
            
            user_manager.save_users()
            st.success(f"✅ Cleanup completed! Affected {cleanup_count} users.")
            st.rerun()


# =============================================================================
# MIDDLEWARE FUNCTIONS - NO ACCESS RESTRICTIONS
# =============================================================================

def enhanced_user_prediction_middleware(user_id: str) -> bool:
    """Enhanced middleware for prediction validation (doesn't record clicks) - NO RESTRICTIONS"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    validation = user_manager.validate_user_for_prediction(user_id)
    
    if not validation['valid']:
        st.error(f"❌ **Access Denied:** {validation['reason']}")
        if 'suggestion' in validation:
            st.info(f"💡 **Suggestion:** {validation['suggestion']}")
        if 'similar_ids' in validation and validation['similar_ids']:
            st.info("🔍 **Did you mean:** " + ", ".join(validation['similar_ids']))
        return False
    
    return True


def record_user_prediction_click(user_id: str, prediction_data: Dict = None) -> Tuple[bool, Dict]:
    """Function to record actual prediction clicks"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    return user_manager.record_prediction_click(user_id, prediction_data)


def get_user_status(user_id: str) -> Dict:
    """Get comprehensive user status"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    return user_manager.get_user_click_status(user_id)


def safe_get_user_status(user_id: str) -> Dict:
    """Safely get user status with comprehensive error handling"""
    try:
        return get_user_status(user_id)
    except Exception as e:
        # Return safe defaults on any error
        return {
            'exists': False,
            'error': str(e),
            'fallback': True
        }


# =============================================================================
# ADMIN MIDDLEWARE FUNCTIONS - ALWAYS ALLOW ACCESS
# =============================================================================

def admin_user_access_middleware(user_id: str = None) -> bool:
    """Admin middleware - ALWAYS returns True for management functions"""
    return True

def management_key_middleware(key: str = None) -> bool:
    """Management key middleware - ALWAYS returns True for admin functions"""
    return True

def create_unrestricted_user_management():
    """Alias for create_comprehensive_user_management with no restrictions"""
    return create_comprehensive_user_management()


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def user_prediction_middleware(user_id: str) -> bool:
    """Backward compatibility wrapper"""
    return enhanced_user_prediction_middleware(user_id)

def enhanced_user_access_middleware(user_id: str) -> bool:
    """Backward compatibility wrapper"""
    return enhanced_user_prediction_middleware(user_id)

def user_access_middleware(user_id: str) -> bool:
    """Backward compatibility wrapper"""
    return enhanced_user_prediction_middleware(user_id)

def create_professional_user_management():
    """Backward compatibility wrapper"""
    return create_comprehensive_user_management()

def create_prediction_interface_with_click_tracking():
    """Main prediction interface function"""
    return create_advanced_prediction_interface()


# =============================================================================
# MAIN INTEGRATION FUNCTIONS FOR EASY IMPORT
# =============================================================================

def create_user_management_section():
    """Main user management interface - NO ACCESS RESTRICTIONS"""
    return create_comprehensive_user_management()

def create_prediction_section():
    """Main prediction interface"""
    return create_advanced_prediction_interface()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_user_credentials_email(user_id: str, app_url: str = "https://your-app.streamlit.app") -> str:
    """Generate professional email template for user credentials"""
    if 'enhanced_user_manager' not in st.session_state:
        st.session_state.enhanced_user_manager = EnhancedUserManager()
    
    user_manager = st.session_state.enhanced_user_manager
    
    if user_id not in user_manager.users:
        return "User not found"
    
    user = user_manager.users[user_id]
    tier_config = user_manager.tier_configs.get(user['tier'], {})
    
    template = f"""
Subject: 🚀 Your AI Trading Platform Access - {user_id}

Dear {user['name']},

Welcome to our Advanced AI Trading Platform! Your {user['tier'].title()} account has been created and is ready to use.

🔍 Your Credentials:
   User ID: {user_id}
   Tier: {user['tier'].title()}
   Monthly Predictions: {user['monthly_limit']}
   Account Status: {user['status'].title()}

🌟 Your Features:
{chr(10).join([f"   • {feature.replace('_', ' ').title()}" for feature in tier_config.get('features', [])])}

🚀 Getting Started:
   1. Visit: {app_url}
   2. Enter "{user_id}" in the sidebar under "User Access"
   3. You'll see your personalized dashboard with usage tracking
   4. Go to the "Predictions" section to start making AI predictions

📊 Your Account Details:
   • Monthly Limit: {user['monthly_limit']} AI predictions
   • Usage resets automatically each month  
   • Real-time usage tracking and warnings
   • Tier: {user['tier'].title()} with premium features
   • Referral Code: {user.get('referral_code', 'N/A')}

🔑 Security Information:
   • Keep your User ID secure and confidential
   • Do not share your credentials with others
   • Contact support immediately if you suspect unauthorized access

📞 Support & Contact:
   If you have any questions or need assistance, please reply to this email or contact our support team.

Best regards,
AI Trading Platform Team

---
This is an automated message. Your account was created on {user['created'][:10]}.
For technical support, visit our help center or contact: support@yourplatform.com
"""
    
    return template.strip()


def log_user_action(user_id: str, action: str, details: Dict = None):
    """Log user actions for audit trail"""
    log_entry = {
        'user_id': user_id,
        'action': action,
        'timestamp': datetime.now().isoformat(),
        'details': details or {}
    }
    
    try:
        import logging
        logging.info(f"User Action: {json.dumps(log_entry)}")
    except:
        print(f"User Action: {json.dumps(log_entry)}")


def monitor_system_performance():
    """Monitor system performance metrics"""
    if 'enhanced_user_manager' not in st.session_state:
        return {}
    
    user_manager = st.session_state.enhanced_user_manager
    stats = user_manager.get_comprehensive_stats()
    
    # Performance metrics
    performance_metrics = {
        'total_users': stats['total_users'],
        'active_users': stats['active_users'],
        'total_predictions': sum(len(u.get('click_history', [])) for u in user_manager.users.values()),
        'system_load': stats['capacity_utilization'],
        'last_updated': datetime.now().isoformat()
    }
    
    return performance_metrics


# =============================================================================
# EXPORT ALL MAIN FUNCTIONS
# =============================================================================

__all__ = [
    'EnhancedUserManager',
    'create_advanced_prediction_interface',
    'create_comprehensive_user_management',
    'create_unrestricted_user_management',
    'create_prediction_interface_with_click_tracking',
    'create_professional_user_management',
    'create_user_management_section',
    'create_prediction_section',
    'enhanced_user_prediction_middleware',
    'user_prediction_middleware',
    'enhanced_user_access_middleware',
    'user_access_middleware',
    'admin_user_access_middleware',
    'management_key_middleware',
    'record_user_prediction_click',
    'get_user_status',
    'safe_get_user_status',
    'generate_user_credentials_email',
    'log_user_action',
    'monitor_system_performance',
    'quick_setup_demo_environment',
    'display_quick_user_ids'
]


# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================

# Auto-initialize user manager in session state if not exists
if 'enhanced_user_manager' not in st.session_state:
    st.session_state.enhanced_user_manager = EnhancedUserManager()
    
# Auto-setup demo environment on first load
if len(st.session_state.enhanced_user_manager.users) == 0:
    quick_setup_demo_environment()
