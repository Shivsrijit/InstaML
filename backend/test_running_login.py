import requests

def test_running():
    url = "http://127.0.0.1:8000/api"
    username = "tempuser123"
    email = "tempuser123@example.com"
    password = "temppass123"
    
    # 1. Register
    print("Registering...")
    res = requests.post(f"{url}/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    print("Register response:", res.status_code, res.text)
    
    # 2. Login with exact casing
    print("\nLogin with exact casing...")
    res = requests.post(f"{url}/auth/login", json={
        "username": username,
        "password": password
    })
    print("Login response:", res.status_code, res.text)
    
    # 3. Login with mixed casing (should fail on old running code, succeed on new)
    print("\nLogin with mixed casing...")
    res = requests.post(f"{url}/auth/login", json={
        "username": "TEMPuser123",
        "password": password
    })
    print("Login response:", res.status_code, res.text)
    
    # 4. Login with email (should fail on old running code, succeed on new)
    print("\nLogin with email...")
    res = requests.post(f"{url}/auth/login", json={
        "username": email,
        "password": password
    })
    print("Login response:", res.status_code, res.text)
    
    # Cleanup
    import sqlite3
    import os
    # Delete from Turso directly
    import sys
    sys.path.insert(0, r"c:\Users\SSN\OneDrive - Shiv Nadar University - Chennai\Desktop\instaml")
    from backend.app.db.session import execute_sync
    try:
        execute_sync("DELETE FROM users WHERE username = ?", [username])
        print("Cleaned up tempuser123 from Turso.")
    except Exception as e:
        print("Error cleaning up from Turso:", e)

if __name__ == "__main__":
    test_running()
