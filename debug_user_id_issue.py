#!/usr/bin/env python3
"""
Debug script to find why User ID is not found
"""

import os
import sqlite3
from shared.user_database import UserDatabase

def debug_user_database():
    """Debug the user database issue"""
    
    print("🔍 DEBUGGING USER DATABASE ISSUE")
    print("=" * 50)
    
    # Step 1: Check what database file is being used
    user_db = UserDatabase()
    print(f"📁 Database file path: {user_db.db_path}")
    print(f"📁 Database file exists: {os.path.exists(user_db.db_path)}")
    
    if os.path.exists(user_db.db_path):
        file_size = os.path.getsize(user_db.db_path)
        print(f"📁 Database file size: {file_size} bytes")
    
    # Step 2: Check what's in the database
    try:
        all_users = user_db.get_all_users()
        print(f"\n👥 Total users in database: {len(all_users)}")
        
        if all_users:
            print("\n📋 Users found:")
            for i, user in enumerate(all_users[:10]):  # Show first 10
                print(f"  {i+1}. {user['user_id']} | {user['tier']} | Active: {user['is_active']}")
        else:
            print("⚠️  No users found in database!")
            
    except Exception as e:
        print(f"❌ Error reading users: {e}")
    
    # Step 3: Check database tables
    try:
        with sqlite3.connect(user_db.db_path) as conn:
            cursor = conn.cursor()
            
            # List all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"\n📊 Tables in database: {[table[0] for table in tables]}")
            
            # Check users table structure
            if any('users' in table for table in tables):
                cursor.execute("PRAGMA table_info(users)")
                columns = cursor.fetchall()
                print(f"\n🏗️  Users table columns: {[col[1] for col in columns]}")
                
                # Count users
                cursor.execute("SELECT COUNT(*) FROM users")
                count = cursor.fetchone()[0]
                print(f"📊 Users table row count: {count}")
                
                # Show sample users
                cursor.execute("SELECT user_id, tier, is_active FROM users LIMIT 5")
                sample_users = cursor.fetchall()
                print(f"📋 Sample users: {sample_users}")
                
    except Exception as e:
        print(f"❌ Error checking database structure: {e}")
    
    # Step 4: Test creating a user
    print("\n🧪 TESTING USER CREATION:")
    test_user_id = "TEST-DEBUG001"
    
    try:
        # Try to create a test user
        success = user_db.create_user(test_user_id, 'tier_10', created_by='debug_test')
        print(f"✅ Test user creation: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            # Try to retrieve the test user
            test_user = user_db.get_user(test_user_id)
            print(f"✅ Test user retrieval: {'SUCCESS' if test_user else 'FAILED'}")
            
            if test_user:
                print(f"📋 Test user details: {test_user['user_id']} | {test_user['tier']}")
            
            # Clean up test user
            try:
                with sqlite3.connect(user_db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM users WHERE user_id = ?", (test_user_id,))
                    conn.commit()
                    print("🧹 Test user cleaned up")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Test user creation failed: {e}")
    
    # Step 5: Check working directory
    print(f"\n📂 Current working directory: {os.getcwd()}")
    print(f"📂 Admin repo files: {[f for f in os.listdir('.') if f.endswith('.db')]}")
    
    # Step 6: Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Check if both admin and user repos are in same directory")
    print("2. Verify database file path is the same in both apps")
    print("3. Make sure you're running both apps from correct directories")
    print("4. Check if there are multiple .db files")
    
    return user_db

def test_specific_user_id(user_id):
    """Test a specific user ID"""
    print(f"\n🔍 TESTING SPECIFIC USER ID: {user_id}")
    print("-" * 40)
    
    user_db = UserDatabase()
    
    # Check if user exists
    user = user_db.get_user(user_id)
    
    if user:
        print(f"✅ User found!")
        print(f"   ID: {user['user_id']}")
        print(f"   Tier: {user['tier']}")
        print(f"   Active: {user['is_active']}")
        print(f"   Predictions: {user['predictions_remaining']}/{user['max_predictions']}")
    else:
        print(f"❌ User NOT found!")
        
        # Check if similar users exist
        all_users = user_db.get_all_users()
        similar = [u['user_id'] for u in all_users if user_id.upper() in u['user_id'].upper()]
        
        if similar:
            print(f"🔍 Similar user IDs found: {similar}")
        else:
            print("🔍 No similar user IDs found")

if __name__ == "__main__":
    # Run debug
    user_db = debug_user_database()
    
    # Test with a specific user ID if provided
    print("\n" + "="*50)
    test_user_input = input("Enter a User ID to test (or press Enter to skip): ").strip()
    
    if test_user_input:
        test_specific_user_id(test_user_input)
    
    print("\n✅ Debug complete!")
