import sqlite3
import hashlib
import secrets
import streamlit as st
from typing import Optional
from datetime import datetime

class UserAuthManager:
    def __init__(self, db_path='users.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_users_table()

    def create_users_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT UNIQUE,
                tier TEXT DEFAULT 'free',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        self.conn.commit()

    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        if salt is None:
            salt = secrets.token_hex(16)
        
        hashed = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        ).hex()
        
        return hashed, salt

    def register_user(self, username: str, password: str, email: str, tier: str = 'free'):
        try:
            if not username or not password or not email:
                raise ValueError("All fields are required")

            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                raise ValueError("Username or email already exists")

            hashed_password, salt = self.hash_password(password)

            cursor.execute('''
                INSERT INTO users 
                (username, hashed_password, salt, email, tier) 
                VALUES (?, ?, ?, ?, ?)
            ''', (username, hashed_password, salt, email, tier))
            
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Registration error: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if not user:
                st.error("User not found")
                return None

            user_id, db_username, db_hashed_password, db_salt, email, tier, *_ = user

            hashed_input, _ = self.hash_password(password, db_salt)
            
            if hashed_input != db_hashed_password:
                st.error("Invalid password")
                return None

            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
            self.conn.commit()

            return {
                'id': user_id,
                'username': db_username,
                'email': email,
                'tier': tier
            }
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return None

    def close(self):
        self.conn.close()

def login_page(auth_manager):
    st.title("Login to AI Trading Professional")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = auth_manager.authenticate_user(username, password)
        if user:
            st.session_state.user = user
            st.success("Login successful!")
            st.experimental_rerun()

def register_page(auth_manager):
    st.title("Register for AI Trading Professional")
    new_username = st.text_input("Choose a Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match")
        else:
            success = auth_manager.register_user(new_username, new_password, new_email)
            if success:
                st.success("Registration successful! Please log in.")