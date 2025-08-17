import sqlite3
import os
from datetime import datetime, timedelta
import uuid

class UserDatabase:
    def __init__(self, db_path='user_management.db'):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                max_predictions INTEGER NOT NULL,
                predictions_used INTEGER NOT NULL,
                predictions_remaining INTEGER NOT NULL,
                premium_key TEXT,
                created_at TEXT NOT NULL,
                last_used TEXT,
                reset_date TEXT NOT NULL,
                is_active BOOLEAN NOT NULL
            )
            ''')
            conn.commit()
    
    def add_user(self, user_details):
        """Add or update a user in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, tier, max_predictions, predictions_used, 
             predictions_remaining, premium_key, created_at, 
             last_used, reset_date, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_details['user_id'], 
                user_details['tier'], 
                user_details['max_predictions'],
                user_details['predictions_used'],
                user_details['predictions_remaining'],
                user_details.get('premium_key', None),
                user_details['created_at'],
                user_details.get('last_used', None),
                user_details['reset_date'],
                user_details['is_active']
            ))
            conn.commit()
    
    def get_user(self, user_id):
        """Retrieve a user's details"""
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
                    'is_active': bool(user[9])
                }
            return None
    
    def use_prediction(self, user_id):
        """Use a prediction and update remaining predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fetch current user details
            cursor.execute('''
            SELECT predictions_used, predictions_remaining, is_active 
            FROM users 
            WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result or not result[2]:  # User not found or inactive
                return False
            
            predictions_used, predictions_remaining, is_active = result
            
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
            
            conn.commit()
            return True
    
    def reset_user_predictions(self, user_id):
        """Reset user predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fetch current max predictions
            cursor.execute('''
            SELECT max_predictions FROM users 
            WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return False
            
            max_predictions = result[0]
            new_reset_date = (datetime.now() + timedelta(days=30)).isoformat()
            
            cursor.execute('''
            UPDATE users 
            SET predictions_used = 0,
                predictions_remaining = ?,
                reset_date = ?
            WHERE user_id = ?
            ''', (max_predictions, new_reset_date, user_id))
            
            conn.commit()
            return True