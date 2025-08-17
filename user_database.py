import sqlite3
import os
from datetime import datetime, timedelta
import uuid
import logging

class UserDatabase:
    def __init__(self, db_path='user_management.db'):
        """
        Initialize the UserDatabase with configurable database path
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Ensure the database is created in the script's directory
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Create tables and perform any necessary migrations
        self._create_tables()
        self._perform_database_migrations()
    
    def _setup_logger(self):
        """
        Set up logging for database operations
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('UserDatabase')
        logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, 'user_database.log'), mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
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
        Create database tables with comprehensive schema
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table with extended attributes
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    max_predictions INTEGER NOT NULL,
                    predictions_used INTEGER NOT NULL,
                    predictions_remaining INTEGER NOT NULL,
                    premium_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    reset_date TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    last_login TEXT,
                    email TEXT,
                    additional_metadata TEXT
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
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                ''')
                
                # Tier configurations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS tier_configs (
                    tier TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    max_predictions INTEGER NOT NULL,
                    features TEXT
                )
                ''')
                
                # Populate tier configurations if not exists
                cursor.execute('''
                INSERT OR IGNORE INTO tier_configs 
                (tier, display_name, max_predictions, features) VALUES 
                ('free', 'Free Tier', 0, '{}'),
                ('tier_10', '10 Predictions Tier', 10, '{"advanced_models": false}'),
                ('tier_25', '25 Predictions Tier', 25, '{"advanced_models": true}'),
                ('tier_50', '50 Predictions Tier', 50, '{"advanced_models": true, "backtesting": true}'),
                ('tier_100', '100 Predictions Tier', 100, '{"advanced_models": true, "backtesting": true, "portfolio_optimization": true}')
                ''')
                
                conn.commit()
                self.logger.info("Database tables created successfully")
        
        except sqlite3.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def _perform_database_migrations(self):
        """
        Perform any necessary database schema migrations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Example migration: Add a new column if it doesn't exist
                cursor.execute('''
                PRAGMA table_info(users)
                ''')
                columns = [column[1] for column in cursor.fetchall()]
                
                migrations = {
                    'email': 'ALTER TABLE users ADD COLUMN email TEXT',
                    'additional_metadata': 'ALTER TABLE users ADD COLUMN additional_metadata TEXT'
                }
                
                for column, migration in migrations.items():
                    if column not in columns:
                        try:
                            cursor.execute(migration)
                            self.logger.info(f"Added column {column} to users table")
                        except sqlite3.OperationalError:
                            self.logger.warning(f"Migration for {column} already applied or failed")
                
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Database migration error: {e}")
    
    def add_user(self, user_details):
        """
        Add or update a user in the database with comprehensive details
        
        Args:
            user_details (dict): Dictionary containing user information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare user details with defaults
                default_details = {
                    'user_id': None,
                    'tier': 'free',
                    'max_predictions': 0,
                    'predictions_used': 0,
                    'predictions_remaining': 0,
                    'premium_key': None,
                    'created_at': datetime.now().isoformat(),
                    'last_used': None,
                    'reset_date': (datetime.now() + timedelta(days=30)).isoformat(),
                    'is_active': True,
                    'last_login': None,
                    'email': None,
                    'additional_metadata': None
                }
                
                # Update default details with provided details
                default_details.update(user_details)
                
                # Upsert operation
                cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, tier, max_predictions, predictions_used, 
                 predictions_remaining, premium_key, created_at, 
                 last_used, reset_date, is_active, last_login, email, additional_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    default_details['user_id'], 
                    default_details['tier'], 
                    default_details['max_predictions'],
                    default_details['predictions_used'],
                    default_details['predictions_remaining'],
                    default_details['premium_key'],
                    default_details['created_at'],
                    default_details['last_used'],
                    default_details['reset_date'],
                    default_details['is_active'],
                    default_details['last_login'],
                    default_details['email'],
                    default_details['additional_metadata']
                ))
                
                # Log user creation/update
                self.logger.info(f"User {default_details['user_id']} added/updated in database")
                
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Error adding/updating user: {e}")
            raise
    
    def get_user(self, user_id):
        """
        Retrieve a user's details
        
        Args:
            user_id (str): User ID to retrieve
        
        Returns:
            dict or None: User details or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
                user = cursor.fetchone()
                
                if user:
                    return {
                        'user_id': user[0],
                        'tier': user[1],
                        'max_predictions': user[2],
                        'predictions_used': user[3],
                        'predictions_remaining': user[4],
                        'premium_key': user[5],
                        'created_at': user[6],
                        'last_used': user[7],
                        'reset_date': user[8],
                        'is_active': bool(user[9]),
                        'last_login': user[10],
                        'email': user[11],
                        'additional_metadata': user[12]
                    }
                return None
        
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving user: {e}")
            return None
    
    def use_prediction(self, user_id):
        """
        Use a prediction and update remaining predictions
        
        Args:
            user_id (str): User ID
        
        Returns:
            bool: True if prediction used, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Fetch current user details
                cursor.execute('''
                SELECT predictions_used, predictions_remaining, is_active, max_predictions 
                FROM users 
                WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                
                if not result or not result[2]:  # User not found or inactive
                    return False
                
                predictions_used, predictions_remaining, is_active, max_predictions = result
                
                if predictions_remaining <= 0:
                    return False
                
                # Update predictions
                cursor.execute('''
                UPDATE users 
                SET predictions_used = ?, 
                    predictions_remaining = ?, 
                    last_used = ?
                WHERE user_id = ?
                ''', (
                    predictions_used + 1, 
                    predictions_remaining - 1, 
                    datetime.now().isoformat(), 
                    user_id
                ))
                
                # Log prediction usage
                cursor.execute('''
                INSERT INTO usage_tracking 
                (user_id, action_type, timestamp, details)
                VALUES (?, ?, ?, ?)
                ''', (
                    user_id, 
                    'prediction_used', 
                    datetime.now().isoformat(), 
                    f'Used prediction. Remaining: {predictions_remaining - 1}'
                ))
                
                conn.commit()
                return True
        
        except sqlite3.Error as e:
            self.logger.error(f"Error using prediction: {e}")
            return False
    
    def reset_user_predictions(self, user_id):
        """
        Reset user predictions
        
        Args:
            user_id (str): User ID
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Fetch max predictions
                cursor.execute('''
                SELECT max_predictions FROM users 
                WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                max_predictions = result[0]
                new_reset_date = (datetime.now() + timedelta(days=30)).isoformat()
                
                # Reset predictions
                cursor.execute('''
                UPDATE users 
                SET predictions_used = 0,
                    predictions_remaining = ?,
                    reset_date = ?
                WHERE user_id = ?
                ''', (max_predictions, new_reset_date, user_id))
                
                # Log reset action
                cursor.execute('''
                INSERT INTO usage_tracking 
                (user_id, action_type, timestamp, details)
                VALUES (?, ?, ?, ?)
                ''', (
                    user_id, 
                    'predictions_reset', 
                    datetime.now().isoformat(), 
                    f'Reset predictions to {max_predictions}'
                ))
                
                conn.commit()
                return True
        
        except sqlite3.Error as e:
            self.logger.error(f"Error resetting predictions: {e}")
            return False
    
    def get_usage_history(self, user_id, limit=10):
        """
        Retrieve usage history for a user
        
        Args:
            user_id (str): User ID
            limit (int): Number of records to retrieve
        
        Returns:
            list: List of usage tracking records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT action_type, timestamp, details 
                FROM usage_tracking 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                ''', (user_id, limit))
                
                return [
                    {
                        'action_type': row[0],
                        'timestamp': row[1],
                        'details': row[2]
                    }
                    for row in cursor.fetchall()
                ]
        
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving usage history: {e}")
            return []
