#!/usr/bin/env python3
"""
Synchronized User Database Management System
Handles user management, prediction tracking, and tier-based access control
Compatible with both admin and user applications
"""

import sqlite3
import os
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

class UserDatabase:
    """
    Centralized database management for user accounts, predictions, and access control
    """
    
    def __init__(self, db_path: str = 'user_management.db'):
        """
        Initialize the UserDatabase with configurable database path
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Ensure database is in consistent location
        self.db_path = os.path.abspath(db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Initialize database
        self._create_tables()
        self._perform_migrations()
        self._initialize_tier_configs()
        
        self.logger.info(f"UserDatabase initialized with path: {self.db_path}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging for database operations
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('UserDatabase')
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        # Create log directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'user_database.log', mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Less verbose in console
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_tables(self):
        """
        Create comprehensive database schema
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute('PRAGMA foreign_keys = ON')
                
                # Users table - Core user management
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'free',
                    max_predictions INTEGER NOT NULL DEFAULT 0,
                    predictions_used INTEGER NOT NULL DEFAULT 0,
                    predictions_remaining INTEGER NOT NULL DEFAULT 0,
                    premium_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    reset_date TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    last_login TEXT,
                    email TEXT,
                    metadata TEXT,
                    created_by TEXT DEFAULT 'system',
                    updated_at TEXT
                )
                ''')
                
                # Usage tracking table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    session_id TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
                ''')
                
                # Tier configurations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS tier_configs (
                    tier TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    max_predictions INTEGER NOT NULL,
                    features TEXT,
                    price REAL DEFAULT 0.0,
                    duration_days INTEGER DEFAULT 30,
                    is_active BOOLEAN NOT NULL DEFAULT 1
                )
                ''')
                
                # Premium keys table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS premium_keys (
                    key_id TEXT PRIMARY KEY,
                    premium_key TEXT UNIQUE NOT NULL,
                    tier TEXT NOT NULL,
                    max_predictions INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    is_used BOOLEAN NOT NULL DEFAULT 0,
                    used_by TEXT,
                    used_at TEXT,
                    created_by TEXT DEFAULT 'admin',
                    FOREIGN KEY(tier) REFERENCES tier_configs(tier),
                    FOREIGN KEY(used_by) REFERENCES users(user_id)
                )
                ''')
                
                # Admin users table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    is_locked BOOLEAN NOT NULL DEFAULT 0,
                    last_login TEXT,
                    failed_attempts INTEGER DEFAULT 0,
                    lockout_time TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT DEFAULT 'system'
                )
                ''')
                
                # Session management table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_tracking(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_tracking(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_premium_keys_used ON premium_keys(is_used)')
                
                conn.commit()
                self.logger.info("Database tables created successfully")
        
        except sqlite3.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def _perform_migrations(self):
        """
        Perform database schema migrations for version compatibility
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current table schema
                cursor.execute('PRAGMA table_info(users)')
                columns = {column[1]: column[2] for column in cursor.fetchall()}
                
                # Define migrations
                migrations = [
                    ('metadata', 'ALTER TABLE users ADD COLUMN metadata TEXT'),
                    ('created_by', 'ALTER TABLE users ADD COLUMN created_by TEXT DEFAULT "system"'),
                    ('updated_at', 'ALTER TABLE users ADD COLUMN updated_at TEXT'),
                ]
                
                for column, migration_sql in migrations:
                    if column not in columns:
                        try:
                            cursor.execute(migration_sql)
                            self.logger.info(f"Applied migration: {column}")
                        except sqlite3.OperationalError as e:
                            self.logger.warning(f"Migration {column} failed or already applied: {e}")
                
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Migration error: {e}")
    
    def _initialize_tier_configs(self):
        """
        Initialize default tier configurations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                default_tiers = [
                    ('free', 'Free Tier', 0, '{"models": 2, "features": ["basic_viewing"]}', 0.0, 30),
                    ('tier_10', '10 Predictions Tier', 10, '{"models": 4, "features": ["predictions", "basic_analytics"]}', 19.99, 30),
                    ('tier_25', '25 Predictions Tier', 25, '{"models": 6, "features": ["predictions", "analytics", "explanations"]}', 39.99, 30),
                    ('tier_50', '50 Predictions Tier', 50, '{"models": 8, "features": ["predictions", "analytics", "explanations", "backtesting"]}', 69.99, 30),
                    ('tier_100', '100 Predictions Tier', 100, '{"models": 8, "features": ["predictions", "analytics", "explanations", "backtesting", "portfolio"]}', 119.99, 30),
                ]
                
                for tier_data in default_tiers:
                    cursor.execute('''
                    INSERT OR IGNORE INTO tier_configs 
                    (tier, display_name, max_predictions, features, price, duration_days, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ''', tier_data)
                
                conn.commit()
                self.logger.info("Tier configurations initialized")
        
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing tier configs: {e}")
    
    def create_user(self, user_id: str, tier: str = 'free', premium_key: Optional[str] = None, 
                   created_by: str = 'system', **kwargs) -> bool:
        """
        Create a new user with comprehensive details
        
        Args:
            user_id (str): Unique user identifier
            tier (str): User tier (free, tier_10, tier_25, tier_50, tier_100)
            premium_key (Optional[str]): Premium key if applicable
            created_by (str): Who created this user
            **kwargs: Additional user attributes
        
        Returns:
            bool: True if user created successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get tier configuration
                cursor.execute('SELECT max_predictions FROM tier_configs WHERE tier = ?', (tier,))
                tier_result = cursor.fetchone()
                
                if not tier_result:
                    self.logger.error(f"Invalid tier: {tier}")
                    return False
                
                max_predictions = tier_result[0]
                
                # Prepare user data
                now = datetime.now().isoformat()
                reset_date = (datetime.now() + timedelta(days=30)).isoformat()
                
                user_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'max_predictions': max_predictions,
                    'predictions_used': 0,
                    'predictions_remaining': max_predictions,
                    'premium_key': premium_key,
                    'created_at': now,
                    'last_used': None,
                    'reset_date': reset_date,
                    'is_active': True,
                    'last_login': None,
                    'email': kwargs.get('email'),
                    'metadata': json.dumps(kwargs.get('metadata', {})),
                    'created_by': created_by,
                    'updated_at': now
                }
                
                # Insert user
                cursor.execute('''
                INSERT INTO users 
                (user_id, tier, max_predictions, predictions_used, predictions_remaining,
                 premium_key, created_at, last_used, reset_date, is_active, last_login,
                 email, metadata, created_by, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(user_data.values()))
                
                # Log user creation
                self._log_action(user_id, 'user_created', f'User created with tier {tier}', created_by)
                
                conn.commit()
                self.logger.info(f"User {user_id} created successfully with tier {tier}")
                return True
        
        except sqlite3.IntegrityError as e:
            self.logger.error(f"User {user_id} already exists: {e}")
            return False
        except sqlite3.Error as e:
            self.logger.error(f"Error creating user {user_id}: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive user details
        
        Args:
            user_id (str): User ID to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: User details or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT u.*, tc.display_name as tier_display_name, tc.features as tier_features
                FROM users u
                LEFT JOIN tier_configs tc ON u.tier = tc.tier
                WHERE u.user_id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Map to dictionary
                columns = [desc[0] for desc in cursor.description]
                user_data = dict(zip(columns, row))
                
                # Parse JSON fields
                if user_data.get('metadata'):
                    try:
                        user_data['metadata'] = json.loads(user_data['metadata'])
                    except json.JSONDecodeError:
                        user_data['metadata'] = {}
                
                if user_data.get('tier_features'):
                    try:
                        user_data['tier_features'] = json.loads(user_data['tier_features'])
                    except json.JSONDecodeError:
                        user_data['tier_features'] = {}
                
                # Calculate next reset date if needed
                if user_data.get('reset_date'):
                    try:
                        reset_dt = datetime.fromisoformat(user_data['reset_date'])
                        if reset_dt < datetime.now():
                            # Auto-reset if past due
                            self.reset_user_predictions(user_id)
                            user_data['predictions_remaining'] = user_data['max_predictions']
                            user_data['predictions_used'] = 0
                    except ValueError:
                        pass
                
                return user_data
        
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    
    def use_prediction(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """
        Use a prediction and update remaining count
        
        Args:
            user_id (str): User ID
            session_id (Optional[str]): Session identifier for tracking
        
        Returns:
            bool: True if prediction used successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current user status
                cursor.execute('''
                SELECT predictions_used, predictions_remaining, is_active, max_predictions, tier
                FROM users 
                WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    self.logger.warning(f"User {user_id} not found for prediction use")
                    return False
                
                predictions_used, predictions_remaining, is_active, max_predictions, tier = result
                
                # Validate user can use prediction
                if not is_active:
                    self.logger.warning(f"User {user_id} is inactive")
                    return False
                
                if predictions_remaining <= 0:
                    self.logger.warning(f"User {user_id} has no predictions remaining")
                    return False
                
                # Update predictions
                new_used = predictions_used + 1
                new_remaining = predictions_remaining - 1
                now = datetime.now().isoformat()
                
                cursor.execute('''
                UPDATE users 
                SET predictions_used = ?, 
                    predictions_remaining = ?, 
                    last_used = ?,
                    updated_at = ?
                WHERE user_id = ?
                ''', (new_used, new_remaining, now, now, user_id))
                
                # Log prediction usage
                self._log_action(
                    user_id, 
                    'prediction_used', 
                    f'Prediction used. Remaining: {new_remaining}/{max_predictions}',
                    session_id=session_id
                )
                
                conn.commit()
                self.logger.info(f"Prediction used for user {user_id}. Remaining: {new_remaining}")
                return True
        
        except sqlite3.Error as e:
            self.logger.error(f"Error using prediction for user {user_id}: {e}")
            return False
    
    def reset_user_predictions(self, user_id: str, admin_user: str = 'system') -> bool:
        """
        Reset user predictions to maximum allowed
        
        Args:
            user_id (str): User ID to reset
            admin_user (str): Admin performing the reset
        
        Returns:
            bool: True if reset successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get max predictions for user's tier
                cursor.execute('''
                SELECT u.max_predictions, u.tier
                FROM users u
                WHERE u.user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                max_predictions, tier = result
                now = datetime.now().isoformat()
                new_reset_date = (datetime.now() + timedelta(days=30)).isoformat()
                
                # Reset predictions
                cursor.execute('''
                UPDATE users 
                SET predictions_used = 0,
                    predictions_remaining = ?,
                    reset_date = ?,
                    updated_at = ?
                WHERE user_id = ?
                ''', (max_predictions, new_reset_date, now, user_id))
                
                # Log reset action
                self._log_action(
                    user_id, 
                    'predictions_reset', 
                    f'Predictions reset to {max_predictions} by {admin_user}',
                    admin_user
                )
                
                conn.commit()
                self.logger.info(f"Predictions reset for user {user_id} by {admin_user}")
                return True
        
        except sqlite3.Error as e:
            self.logger.error(f"Error resetting predictions for user {user_id}: {e}")
            return False
    
    def update_user_tier(self, user_id: str, new_tier: str, admin_user: str = 'system') -> bool:
        """
        Update user's tier and adjust predictions accordingly
        
        Args:
            user_id (str): User ID to update
            new_tier (str): New tier to assign
            admin_user (str): Admin performing the update
        
        Returns:
            bool: True if update successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get new tier configuration
                cursor.execute('SELECT max_predictions FROM tier_configs WHERE tier = ?', (new_tier,))
                tier_result = cursor.fetchone()
                
                if not tier_result:
                    self.logger.error(f"Invalid tier: {new_tier}")
                    return False
                
                new_max_predictions = tier_result[0]
                now = datetime.now().isoformat()
                
                # Get current user data
                user = self.get_user(user_id)
                if not user:
                    return False
                
                # Calculate new remaining predictions
                current_used = user['predictions_used']
                new_remaining = max(0, new_max_predictions - current_used)
                
                # Update user
                cursor.execute('''
                UPDATE users 
                SET tier = ?, 
                    max_predictions = ?, 
                    predictions_remaining = ?,
                    updated_at = ?
                WHERE user_id = ?
                ''', (new_tier, new_max_predictions, new_remaining, now, user_id))
                
                # Log tier change
                self._log_action(
                    user_id, 
                    'tier_updated', 
                    f'Tier updated from {user["tier"]} to {new_tier} by {admin_user}',
                    admin_user
                )
                
                conn.commit()
                self.logger.info(f"User {user_id} tier updated to {new_tier}")
                return True
        
        except sqlite3.Error as e:
            self.logger.error(f"Error updating tier for user {user_id}: {e}")
            return False
    
    def create_premium_key(self, tier: str, created_by: str = 'admin', 
                          expires_days: int = 365) -> Optional[str]:
        """
        Generate a premium key for a specific tier
        
        Args:
            tier (str): Tier for the premium key
            created_by (str): Admin creating the key
            expires_days (int): Key expiration in days
        
        Returns:
            Optional[str]: Generated premium key or None if failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Validate tier
                cursor.execute('SELECT max_predictions FROM tier_configs WHERE tier = ?', (tier,))
                tier_result = cursor.fetchone()
                
                if not tier_result:
                    self.logger.error(f"Invalid tier for premium key: {tier}")
                    return None
                
                max_predictions = tier_result[0]
                
                # Generate unique key
                key_id = f"KEY-{uuid.uuid4().hex[:8].upper()}"
                premium_key = f"PREM-{tier.upper()}-{uuid.uuid4().hex[:12].upper()}"
                
                now = datetime.now().isoformat()
                expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
                
                # Insert premium key
                cursor.execute('''
                INSERT INTO premium_keys 
                (key_id, premium_key, tier, max_predictions, created_at, expires_at, 
                 is_used, created_by)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ''', (key_id, premium_key, tier, max_predictions, now, expires_at, created_by))
                
                conn.commit()
                self.logger.info(f"Premium key created: {premium_key} for tier {tier}")
                return premium_key
        
        except sqlite3.Error as e:
            self.logger.error(f"Error creating premium key: {e}")
            return None
    
    def validate_premium_key(self, premium_key: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate a premium key and optionally assign to user
        
        Args:
            premium_key (str): Premium key to validate
            user_id (Optional[str]): User ID to assign key to
        
        Returns:
            Dict[str, Any]: Validation result
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check premium key
                cursor.execute('''
                SELECT pk.tier, pk.max_predictions, pk.is_used, pk.expires_at, pk.used_by
                FROM premium_keys pk
                WHERE pk.premium_key = ?
                ''', (premium_key,))
                
                result = cursor.fetchone()
                
                if not result:
                    return {'valid': False, 'message': 'Premium key not found'}
                
                tier, max_predictions, is_used, expires_at, used_by = result
                
                # Check if already used
                if is_used and used_by != user_id:
                    return {'valid': False, 'message': 'Premium key already used'}
                
                # Check expiration
                if expires_at:
                    try:
                        exp_date = datetime.fromisoformat(expires_at)
                        if exp_date < datetime.now():
                            return {'valid': False, 'message': 'Premium key expired'}
                    except ValueError:
                        pass
                
                # If user_id provided, assign key
                if user_id and not is_used:
                    # Update premium key as used
                    now = datetime.now().isoformat()
                    cursor.execute('''
                    UPDATE premium_keys 
                    SET is_used = 1, used_by = ?, used_at = ?
                    WHERE premium_key = ?
                    ''', (user_id, now, premium_key))
                    
                    # Update user with premium key and tier
                    cursor.execute('''
                    UPDATE users 
                    SET premium_key = ?, tier = ?, max_predictions = ?, 
                        predictions_remaining = ?, updated_at = ?
                    WHERE user_id = ?
                    ''', (premium_key, tier, max_predictions, max_predictions, now, user_id))
                    
                    # Log key usage
                    self._log_action(user_id, 'premium_key_used', f'Premium key activated for tier {tier}')
                    
                    conn.commit()
                
                return {
                    'valid': True,
                    'tier': tier,
                    'max_predictions': max_predictions,
                    'message': f'Premium key valid for {tier}'
                }
        
        except sqlite3.Error as e:
            self.logger.error(f"Error validating premium key: {e}")
            return {'valid': False, 'message': 'Database error during validation'}
    
    def get_all_users(self, include_inactive: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve all users with their details
        
        Args:
            include_inactive (bool): Whether to include inactive users
        
        Returns:
            List[Dict[str, Any]]: List of user details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                SELECT u.*, tc.display_name as tier_display_name
                FROM users u
                LEFT JOIN tier_configs tc ON u.tier = tc.tier
                '''
                
                if not include_inactive:
                    query += ' WHERE u.is_active = 1'
                
                query += ' ORDER BY u.created_at DESC'
                
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                users = []
                
                for row in cursor.fetchall():
                    user_data = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    if user_data.get('metadata'):
                        try:
                            user_data['metadata'] = json.loads(user_data['metadata'])
                        except json.JSONDecodeError:
                            user_data['metadata'] = {}
                    
                    users.append(user_data)
                
                return users
        
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving all users: {e}")
            return []
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM users')
                total_users = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
                active_users = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(predictions_used) FROM users')
                total_predictions_used = cursor.fetchone()[0] or 0
                
                cursor.execute('SELECT SUM(max_predictions) FROM users')
                total_predictions_allocated = cursor.fetchone()[0] or 0
                
                # Tier breakdown
                cursor.execute('''
                SELECT u.tier, tc.display_name, COUNT(*) as user_count,
                       SUM(u.predictions_used) as predictions_used,
                       SUM(u.max_predictions) as predictions_allocated
                FROM users u
                LEFT JOIN tier_configs tc ON u.tier = tc.tier
                GROUP BY u.tier, tc.display_name
                ''')
                
                tier_breakdown = []
                for row in cursor.fetchall():
                    tier_breakdown.append({
                        'tier': row[0],
                        'display_name': row[1],
                        'user_count': row[2],
                        'predictions_used': row[3],
                        'predictions_allocated': row[4]
                    })
                
                # Recent activity
                cursor.execute('''
                SELECT action_type, COUNT(*) as count
                FROM usage_tracking
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY action_type
                ORDER BY count DESC
                ''')
                
                recent_activity = [{'action': row[0], 'count': row[1]} for row in cursor.fetchall()]
                
                return {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_predictions_used': total_predictions_used,
                    'total_predictions_allocated': total_predictions_allocated,
                    'usage_rate': (total_predictions_used / total_predictions_allocated * 100) if total_predictions_allocated > 0 else 0,
                    'tier_breakdown': tier_breakdown,
                    'recent_activity': recent_activity,
                    'timestamp': datetime.now().isoformat()
                }
        
        except sqlite3.Error as e:
            self.logger.error(f"Error getting usage statistics: {e}")
            return {}
    
    def _log_action(self, user_id: str, action_type: str, details: str = '', 
                   actor: str = 'system', session_id: Optional[str] = None):
        """
        Log user actions for tracking and auditing
        
        Args:
            user_id (str): User ID
            action_type (str): Type of action
            details (str): Action details
            actor (str): Who performed the action
            session_id (Optional[str]): Session identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO usage_tracking 
                (user_id, action_type, timestamp, details, session_id)
                VALUES (?, ?, ?, ?, ?)
                ''', (user_id, action_type, datetime.now().isoformat(), 
                      f"{details} (by: {actor})", session_id))
                
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Error logging action: {e}")
    
    def get_user_usage_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get usage history for a specific user
        
        Args:
            user_id (str): User ID
            limit (int): Maximum number of records
        
        Returns:
            List[Dict[str, Any]]: Usage history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT action_type, timestamp, details, session_id
                FROM usage_tracking
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                ''', (user_id, limit))
                
                return [
                    {
                        'action_type': row[0],
                        'timestamp': row[1],
                        'details': row[2],
                        'session_id': row[3]
                    }
                    for row in cursor.fetchall()
                ]
        
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user history: {e}")
            return []
    
    def cleanup_expired_sessions(self):
        """Clean up expired user sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                DELETE FROM user_sessions 
                WHERE expires_at < ? OR is_active = 0
                ''', (datetime.now().isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired sessions")
        
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database
        
        Args:
            backup_path (Optional[str]): Custom backup path
        
        Returns:
            str: Path to backup file
        """
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"user_management_backup_{timestamp}.db"
        
        try:
            # Create backup using SQLite backup API
            source = sqlite3.connect(self.db_path)
            backup = sqlite3.connect(backup_path)
            
            source.backup(backup)
            
            source.close()
            backup.close()
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        
        except sqlite3.Error as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def get_tier_configs(self) -> List[Dict[str, Any]]:
        """
        Get all tier configurations
        
        Returns:
            List[Dict[str, Any]]: Tier configurations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT tier, display_name, max_predictions, features, price, duration_days, is_active
                FROM tier_configs
                WHERE is_active = 1
                ORDER BY max_predictions ASC
                ''')
                
                tiers = []
                for row in cursor.fetchall():
                    tier_data = {
                        'tier': row[0],
                        'display_name': row[1],
                        'max_predictions': row[2],
                        'price': row[4],
                        'duration_days': row[5],
                        'is_active': bool(row[6])
                    }
                    
                    # Parse features JSON
                    if row[3]:
                        try:
                            tier_data['features'] = json.loads(row[3])
                        except json.JSONDecodeError:
                            tier_data['features'] = {}
                    else:
                        tier_data['features'] = {}
                    
                    tiers.append(tier_data)
                
                return tiers
        
        except sqlite3.Error as e:
            self.logger.error(f"Error getting tier configs: {e}")
            return []
    
    def close(self):
        """Close database connections and cleanup"""
        self.logger.info("UserDatabase instance closed")

# Compatibility wrapper for existing code
class DatabaseManager:
    """Backward compatibility wrapper"""
    
    def __init__(self, db_path='user_management.db'):
        self.user_db = UserDatabase(db_path)
    
    def _get_connection(self):
        return sqlite3.connect(self.user_db.db_path)

# Export functions for direct use in other modules
def get_user_database(db_path: str = 'user_management.db') -> UserDatabase:
    """
    Get a UserDatabase instance
    
    Args:
        db_path (str): Database path
    
    Returns:
        UserDatabase: Database instance
    """
    return UserDatabase(db_path)

def create_default_admin(db_path: str = 'user_management.db', 
                        username: str = 'admin', 
                        password: str = 'admin123') -> bool:
    """
    Create default admin user
    
    Args:
        db_path (str): Database path
        username (str): Admin username
        password (str): Admin password
    
    Returns:
        bool: True if created successfully
    """
    try:
        db = UserDatabase(db_path)
        
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if admin exists
            cursor.execute('SELECT COUNT(*) FROM admin_users WHERE username = ?', (username,))
            if cursor.fetchone()[0] > 0:
                return True  # Already exists
            
            # Hash password
            salt = uuid.uuid4().hex
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            ).hex()
            
            # Create admin
            cursor.execute('''
            INSERT INTO admin_users 
            (username, password_hash, salt, role, is_active, created_at)
            VALUES (?, ?, ?, 'admin', 1, ?)
            ''', (username, password_hash, salt, datetime.now().isoformat()))
            
            conn.commit()
            db.logger.info(f"Default admin user created: {username}")
            return True
    
    except Exception as e:
        print(f"Error creating default admin: {e}")
        return False
